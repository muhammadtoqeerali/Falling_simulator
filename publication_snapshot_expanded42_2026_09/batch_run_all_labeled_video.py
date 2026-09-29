# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import signal
import time
import pexpect
from datetime import datetime
from pathlib import Path

from batch_run_all_labeled import (
    parse_task_list,
    parse_latest_output_folder,
    materialize_relative_outputs,
    wait_for_main_csv,
    parse_event_summary,
    parse_validation_quality,
    scenario_id_from_folder,
    label_csv_direct,
    verify_labeled_csv,
    json_safe,
)


def x11_input_from_display(display: str):
    display = str(display).strip()

    if not display:
        return ":2.0"

    # :2 -> :2.0
    if display.startswith(":") and "." not in display:
        return display + ".0"

    return display


def detect_video_size(display: str):
    def even_size(size_text):
        try:
            w, h = size_text.lower().split("x")
            w = int(w)
            h = int(h)
            # H.264/yuv420p requires even width and height.
            w = w - (w % 2)
            h = h - (h % 2)
            return f"{w}x{h}"
        except Exception:
            return "1280x720"

    if not shutil.which("xdpyinfo"):
        return "1280x720"

    try:
        out = subprocess.check_output(
            ["xdpyinfo", "-display", display],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        m = re.search(r"dimensions:\s+(\d+x\d+)\s+pixels", out)
        if m:
            return even_size(m.group(1))
    except Exception:
        pass

    return "1280x720"


def start_video_recording(display: str, video_path: Path, fps: int, video_size: str):
    """
    Robust per-scenario screen recording.

    Fixes:
    - do not keep ffmpeg stderr in an unread PIPE
    - write ffmpeg diagnostics to a sidecar .ffmpeg.log
    - fail visibly if ffmpeg exits immediately
    - use SIGINT during stop so MP4 finalizes cleanly
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install with: sudo apt install -y ffmpeg")

    video_path.parent.mkdir(parents=True, exist_ok=True)

    x11_input = x11_input_from_display(display)
    err_log = video_path.with_suffix(".ffmpeg.log")
    err_file = err_log.open("w", encoding="utf-8", errors="ignore")

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "info",
        "-thread_queue_size",
        "512",
        "-f",
        "x11grab",
        "-draw_mouse",
        "0",
        "-framerate",
        str(fps),
        "-video_size",
        video_size,
        "-i",
        x11_input,
        "-an",
        "-vf",
        "crop=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]

    print("[video] Recording display:", x11_input)
    print("[video] Video temp file :", video_path)
    print("[video] ffmpeg log      :", err_log)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=err_file,
        stderr=err_file,
        text=True,
    )

    # Give ffmpeg time to fail early if x11grab/display is unavailable.
    time.sleep(2.0)

    if proc.poll() is not None:
        err_file.close()
        tail = ""
        try:
            tail = "\n".join(err_log.read_text(errors="ignore").splitlines()[-40:])
        except Exception:
            pass
        raise RuntimeError(
            "ffmpeg exited immediately before recording video.\n"
            f"ffmpeg log: {err_log}\n"
            f"{tail}"
        )

    return {
        "proc": proc,
        "err_file": err_file,
        "err_log": err_log,
        "video_path": video_path,
    }


def stop_video_recording(handle):
    if handle is None:
        return

    proc = handle.get("proc") if isinstance(handle, dict) else handle
    err_file = handle.get("err_file") if isinstance(handle, dict) else None

    try:
        if proc and proc.poll() is None:
            # SIGINT lets ffmpeg write the MP4 trailer/moov atom correctly.
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
    finally:
        if err_file:
            try:
                err_file.close()
            except Exception:
                pass


def run_one_task_with_video(task_id, age, height, sex, weight, log_path, display, tmp_video, fps, video_size):
    """
    Prompt-aware runner.

    The older version wrote all inputs at once. That can hang on some scenarios
    when prompts are printed without clean newlines. This version waits for each
    prompt and answers it explicitly.
    """
    env = os.environ.copy()
    env["DISPLAY"] = display

    print("\n" + "=" * 90)
    print(f"RUNNING SCENARIO {task_id}")
    print("=" * 90)
    print(f"Subject: age={age}, height={height}, sex={sex}, weight={weight}")
    print(f"Log: {log_path}")
    print(f"Video temp: {tmp_video}")
    print("=" * 90 + "\n")

    video_handle = None

    child = None
    chunks = []

    try:
        child = pexpect.spawn(
            sys.executable,
            ["fall_dispatcher.py"],
            env=env,
            encoding="utf-8",
            codec_errors="ignore",
            timeout=600,
            echo=False,
        )

        with log_path.open("w", encoding="utf-8", errors="ignore") as f:

            def drain_before():
                text = child.before or ""
                if text:
                    print(text, end="")
                    f.write(text)
                    f.flush()
                    chunks.append(text)

            def answer(pattern, value, timeout=600):
                child.expect(pattern, timeout=timeout)
                drain_before()
                child.sendline(str(value))

            # Main menu prompt
            answer(r"Select scenario ID.*:", task_id)

            # Subject prompts
            answer(r"Age\s*\[.*?\]\s*:", age)
            answer(r"Height\s*\[.*?\]\s*:", height)
            answer(r"Sex\s*\[.*?\].*?:", sex)
            answer(r"Weight\s*\[.*?\]\s*:", weight)

            # Now read until the scenario fully exits.
            while True:
                try:
                    idx = child.expect([pexpect.EOF, r".+\n"], timeout=1200)

                    if idx == 0:
                        drain_before()
                        break

                    drain_before()
                    line = child.after or ""
                    if line:
                        print(line, end="")
                        f.write(line)
                        f.flush()
                        chunks.append(line)

                except pexpect.TIMEOUT:
                    # Save useful context before failing.
                    partial = child.before or ""
                    if partial:
                        print(partial, end="")
                        f.write(partial)
                        f.flush()
                        chunks.append(partial)
                    raise RuntimeError(
                        "Scenario timed out while running. "
                        f"Last output was:\n{partial[-2000:]}"
                    )

        code = child.exitstatus
        if code is None:
            code = 0 if child.signalstatus is None else child.signalstatus

    finally:
        try:
            if child is not None and child.isalive():
                child.close(force=True)
        except Exception:
            pass

        stop_video_recording(video_handle)

    return code, "".join(chunks)


def write_summary(rows, batch_dir):
    if not rows:
        return

    csv_path = batch_dir / "batch_summary.csv"
    json_path = batch_dir / "batch_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(rows, indent=2, default=json_safe),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age", required=True)
    ap.add_argument("--height", required=True)
    ap.add_argument("--sex", required=True, choices=["male", "female"])
    ap.add_argument("--weight", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":2"))
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--video-size", default="")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    tasks = parse_task_list(args.tasks)
    project_dir = Path.cwd()

    video_size = args.video_size.strip() or detect_video_size(args.display)

    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path("outputs") / "batch_runs" / f"batch_video_{batch_stamp}"
    log_dir = batch_dir / "logs"
    video_tmp_dir = batch_dir / "video_tmp"

    batch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    video_tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\n" + "#" * 90)
    print("BATCH FALL SIMULATION + DIRECT SAFE LABELING + VIDEO")
    print("#" * 90)
    print(f"Tasks      : {tasks}")
    print(f"Subject    : age={args.age}, height={args.height}, sex={args.sex}, weight={args.weight}")
    print(f"Display    : {args.display}")
    print(f"Video size : {video_size}")
    print(f"Video FPS  : {args.video_fps}")
    print(f"Batch folder: {batch_dir}")
    print("#" * 90 + "\n")

    for i, task_id in enumerate(tasks, start=1):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"scenario{task_id}_{stamp}.log"
        tmp_video = video_tmp_dir / f"scenario{task_id}_{stamp}.mp4"

        row = {
            "task_id": task_id,
            "status": "",
            "output_folder": "",
            "main_csv": "",
            "backup_csv": "",
            "metadata_json": "",
            "video_path": "",
            "label_status": "",
            "rows": "",
            "columns": "",
            "impact_label_count": "",
            "perturbation_start_time_s": "",
            "fall_onset_time_s": "",
            "main_impact_time_s": "",
            "settle_time_s": "",
            "event_source": "",
            "overall_confidence_score": "",
            "classification": "",
            "sisfall_compliant": "",
            "kfall_style_compliant": "",
            "task_validity": "",
            "authenticity_score": "",
            "log_path": str(log_path),
            "error": "",
        }

        try:
            code, log_text = run_one_task_with_video(
                task_id=task_id,
                age=args.age,
                height=args.height,
                sex=args.sex,
                weight=args.weight,
                log_path=log_path,
                display=args.display,
                tmp_video=tmp_video,
                fps=args.video_fps,
                video_size=video_size,
            )

            if code != 0:
                raise RuntimeError(f"fall_dispatcher.py exited with code {code}")

            out_dir = parse_latest_output_folder(log_text)

            if out_dir is None:
                raise RuntimeError("Could not detect output folder from log")

            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            row["output_folder"] = str(out_dir)

            moved = materialize_relative_outputs(log_text, out_dir, project_dir)
            main_csv = wait_for_main_csv(out_dir)

            if main_csv is None or not main_csv.exists():
                raise RuntimeError(f"No main IMU CSV found in {out_dir}")

            final_video = out_dir / f"{main_csv.stem}_display.mp4"

            # Prefer native MuJoCo renderer output.
            # This avoids unreliable X11/VNC screen capture.
            native_video = out_dir / "mujoco_native_render.mp4"

            if native_video.exists() and native_video.stat().st_size > 1024:
                shutil.copy2(str(native_video), str(final_video))

                row["video_path"] = str(final_video)

                print("[video] Saved native MuJoCo render:", final_video)

            elif tmp_video.exists() and tmp_video.stat().st_size > 1024:
                shutil.move(str(tmp_video), str(final_video))

                row["video_path"] = str(final_video)

                print("[video] Saved X11 fallback video:", final_video)

            else:
                err_log = tmp_video.with_suffix(".ffmpeg.log")
                tail = ""

                if err_log.exists():
                    try:
                        tail = "\n".join(
                            err_log.read_text(errors="ignore").splitlines()[-60:]
                        )
                    except Exception:
                        tail = ""

                raise RuntimeError(
                    "No usable video was created.\n"
                    f"Native renderer expected: {native_video}\n"
                    f"X11 temp expected: {tmp_video}\n"
                    f"ffmpeg log: {err_log}\n"
                    f"{tail}"
                )

            events = parse_event_summary(log_text)
            quality = parse_validation_quality(log_text)

            print("\n" + "-" * 90)
            print(f"POST-RUN DIRECT LABELING FOR SCENARIO {task_id}")
            print(f"Output folder : {out_dir}")
            print(f"Main IMU CSV  : {main_csv}")
            print(f"Video         : {row['video_path']}")
            print(f"Moved files   : {len(moved)}")
            print(f"Event summary : {events if events else 'NOT FOUND - direct signal fallback'}")
            print("-" * 90)

            result = label_csv_direct(
                csv_path=main_csv,
                output_dir=out_dir,
                scenario_id=scenario_id_from_folder(out_dir) or task_id,
                event_overrides=events,
                subject_params={
                    "age": args.age,
                    "height": args.height,
                    "sex": args.sex,
                    "weight": args.weight,
                },
                log_quality=quality,
            )

            verify = verify_labeled_csv(Path(result["csv"]))

            row.update({
                "status": "completed_saved_labeled_verified",
                "main_csv": result["csv"],
                "backup_csv": result["backup_csv"],
                "metadata_json": result["metadata_json"],
                "label_status": result["status"],
                "rows": verify["rows"],
                "columns": verify["columns"],
                "impact_label_count": verify["impact_label_count"],
                "perturbation_start_time_s": result["events"].get("perturbation_start_time_s", ""),
                "fall_onset_time_s": verify["fall_onset_time_s"],
                "main_impact_time_s": verify["main_impact_time_s"],
                "settle_time_s": result["events"].get("settle_time_s", ""),
                "event_source": result["events"].get("source", ""),
                **quality,
            })

            print("\nSAVE/LABEL/VIDEO VERIFICATION PASSED")
            print(json.dumps(verify, indent=2))
            print("Backup CSV    :", result["backup_csv"])
            print("Metadata JSON :", result["metadata_json"])
            print("Video MP4     :", row["video_path"])

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            print("\n" + "!" * 90)
            print(f"FAILED SCENARIO {task_id}: {exc}")
            print("!" * 90 + "\n")

            rows.append(row)
            write_summary(rows, batch_dir)

            if not args.continue_on_error:
                print("Batch stopped because this scenario was not safely saved/labeled/video-saved.")
                break

            continue

        rows.append(row)
        write_summary(rows, batch_dir)

        print("\n" + "#" * 90)
        print(f"FINISHED {i}/{len(tasks)} SCENARIOS")
        print(f"Summary CSV : {batch_dir / 'batch_summary.csv'}")
        print(f"Summary JSON: {batch_dir / 'batch_summary.json'}")
        print("#" * 90 + "\n")

    print("\n" + "=" * 90)
    print("BATCH ENDED")
    print(f"Summary folder: {batch_dir}")
    print(f"Summary CSV   : {batch_dir / 'batch_summary.csv'}")
    print(f"Summary JSON  : {batch_dir / 'batch_summary.json'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
