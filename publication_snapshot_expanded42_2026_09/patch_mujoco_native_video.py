from pathlib import Path

p = Path("fall_core.py")

text = p.read_text()

# Add cv2 import if missing
if "import cv2" not in text:
    text = text.replace(
        "import numpy as np",
        "import numpy as np\nimport cv2"
    )

# Add renderer initialization
old = """    with viewer_ctx as viewer:
        if scenario.viewer_enabled:
            viewer.cam.distance  = 4.5
            viewer.cam.elevation = -10
            viewer.cam.azimuth   = 90
"""

new = """    with viewer_ctx as viewer:

        native_video_writer = None
        native_renderer = None

        if scenario.viewer_enabled:

            viewer.cam.distance  = 4.5
            viewer.cam.elevation = -10
            viewer.cam.azimuth   = 90

            native_renderer = mujoco.Renderer(
                mj_model,
                height=1008,
                width=1920
            )

            native_video_path = (
                output_dir /
                "mujoco_native_render.mp4"
            )

            native_video_writer = cv2.VideoWriter(
                str(native_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (1920,1008)
            )

            print(
                "[native video] Recording:",
                native_video_path
            )
"""

if old not in text:
    raise RuntimeError("Viewer block not found")

text = text.replace(old,new)


# Replace viewer sync block
old2 = """            if scenario.viewer_enabled:
                viewer.sync()
"""

new2 = """            if scenario.viewer_enabled:

                viewer.sync()

                native_renderer.update_scene(
                    mj_data
                )

                frame = native_renderer.render()

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )

                native_video_writer.write(
                    frame
                )
"""

if old2 not in text:
    raise RuntimeError("viewer.sync block not found")

text = text.replace(old2,new2)


# Add release before return
marker = """    return summary
"""

replacement = """    if native_video_writer is not None:
        native_video_writer.release()

    if native_renderer is not None:
        native_renderer.close()

    return summary
"""

if marker not in text:
    raise RuntimeError("return summary marker not found")

text=text.replace(marker,replacement,1)

p.write_text(text)

print("PATCH COMPLETE")
