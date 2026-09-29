from pathlib import Path
import cv2
import sys


video = Path(sys.argv[1])
out = Path(sys.argv[2])

out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))

idx = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    fname = out / f"frame_{idx:06d}.png"

    cv2.imwrite(
        str(fname),
        frame
    )

    idx += 1

cap.release()

print("Frames extracted:", idx)
print("Saved:", out)
