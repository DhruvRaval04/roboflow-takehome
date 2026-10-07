"""look_at.py -- "look at the <object>": VLM grounding drives the camera pan servo (v1).

    ESP32 MJPEG --> LatestFrame --> Grounder.locate() --> box center --> pan servo (V cmd, USB serial)
                                                     \--> runs/<timestamp>/  (every frame + answer logged)

Pan only, wheels never move (no M commands are sent, so the firmware watchdog
is irrelevant and the car can't drive off the table).

Closed-loop success = target box center within DEADBAND of image center on 2
consecutive inferences ("LOCKED"). Time-to-lock is the real-world metric for
the results table, next to offline IoU -- the two can disagree, and that gap is
the interesting part of the project.

Run (camera on router, Pico on COM7):
    .venv\\Scripts\\python.exe look_at.py mug
    .venv\\Scripts\\python.exe look_at.py mug --no-car       # camera + model only, no serial
    .venv\\Scripts\\python.exe look_at.py mug --pan-sign -1  # if the camera turns AWAY from the target
Keys in the window: q = quit, c = re-center the camera and restart the timer.
"""
import argparse
import json
import time
from pathlib import Path

import cv2

# camera.py (stream reader) and drive_keys.py (serial handshake) are the car-side
# helpers, vendored next to this file from the rc-car project.
from camera import LatestFrame  # noqa: E402

from vlm import Grounder, draw  # noqa: E402

SERVO_CENTER = 135          # camera straight ahead (measured; matches SerialDrive.ino)
SERVO_MIN, SERVO_MAX = 95, 175  # +-40 deg usable range
SERVO_DEG_PER_S = 60        # firmware slews gradually at ~60 deg/s

# Horizontal field of view of the OV3660 + stock lens. NOT yet measured --
# 60 deg is a placeholder. It converts "target is 0.5 half-widths right of
# center" into degrees of pan: delta = err * HFOV/2. Too small -> sluggish,
# too big -> overshoot. Calibrate later: pan 10 deg, measure pixel shift.
HFOV_DEG = 60

# Proportional gain < 1 on purpose: each inference takes ~0.5-1 s, so the
# controller acts on a stale view. Correcting only ~70% per step converges in a
# few steps instead of overshooting and hunting left-right.
GAIN = 0.7

# |error| below this (fraction of half-width) counts as centered. 0.1 of a 160px
# half-width = 16 px -- about the jitter of the model's box edges anyway.
DEADBAND = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("--port", default="COM7")
    ap.add_argument("--no-car", action="store_true")
    ap.add_argument("--pan-sign", type=int, default=1, choices=[1, -1],
                    help="+1 if increasing servo angle pans the camera RIGHT (unverified)")
    ap.add_argument("--model", default="paligemma2-3b-pt-224",
                    help="Roboflow model ID: pretrained alias (zero-shot) or project/version (our SFT/GRPO LoRAs)")
    ap.add_argument("--api-key", default=None, help="Roboflow API key, only for project/version models")
    args = ap.parse_args()

    ser = None
    if not args.no_car:
        from drive_keys import open_car  # PONG handshake proves COM7 is SerialDrive
        ser = open_car(args.port)
        ser.write(b"S\n")  # belt and braces: wheels stopped before anything else
        ser.write(f"V {SERVO_CENTER}\n".encode())

    run_dir = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    (run_dir / "frames").mkdir(parents=True)
    log = open(run_dir / "log.jsonl", "w")

    print("loading model...")
    g = Grounder(args.model, api_key=args.api_key)
    cam = LatestFrame()
    servo, locked_streak, t_start, i = SERVO_CENTER, 0, time.monotonic(), 0
    wait_until, last_id = 0.0, 0
    try:
        while True:
            fid, frame = cam.read()
            # Only infer on a frame captured AFTER the servo finished moving.
            # A frame grabbed mid-pan shows the scene at an unknown angle (and
            # motion-blurred); acting on it double-counts the correction we
            # already sent -> overshoot.
            if frame is None or fid == last_id or time.monotonic() < wait_until:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue
            last_id = fid
            frame = frame.copy()  # LatestFrame's reader thread owns the original buffer

            r = g.locate(frame, args.obj)
            servo_before = servo
            err = None
            if r["box"] is not None:
                x1, _, x2, _ = r["box"]
                # err in [-1, 1]: -1 = left edge, 0 = center, +1 = right edge
                err = ((x1 + x2) / 2 - frame.shape[1] / 2) / (frame.shape[1] / 2)
                if abs(err) > DEADBAND:
                    delta = args.pan_sign * GAIN * err * HFOV_DEG / 2
                    servo = int(round(min(SERVO_MAX, max(SERVO_MIN, servo + delta))))
                    locked_streak = 0
                else:
                    locked_streak += 1
            else:
                locked_streak = 0  # v1: target not visible -> hold still (v2: search sweep)

            if ser and servo != servo_before:
                ser.write(f"V {servo}\n".encode())
                # Settle time = slew time + ~150 ms for the next JPEG to be
                # exposed, encoded and sent at ~7 fps.
                wait_until = time.monotonic() + abs(servo - servo_before) / SERVO_DEG_PER_S + 0.15

            elapsed = time.monotonic() - t_start
            status = "LOCKED" if locked_streak >= 2 else ("searching" if err is None else f"err {err:+.2f}")
            if locked_streak == 2:
                print(f"LOCKED on {args.obj!r} after {elapsed:.1f} s ({i + 1} inferences)")

            # Log EVERYTHING the car sees + what the model said. These frames are
            # free in-domain data: v2 auto-labels and reviews them in Roboflow.
            cv2.imwrite(str(run_dir / "frames" / f"{i:06d}.jpg"), frame)
            log.write(json.dumps({"i": i, "t": round(elapsed, 3), "obj": args.obj, "servo_before": servo_before,
                                  "servo_after": servo, "box": r["box"], "err": err, "ms": round(r["ms"]),
                                  "raw": r["raw"]}) + "\n")
            log.flush()
            print(f"[{i:4d}] {r['ms']:5.0f} ms  servo {servo_before}->{servo}  {status}")

            vis = draw(frame, r["box"], args.obj)
            cv2.putText(vis, f"{status}  {r['ms']:.0f}ms", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow("look_at (q quit, c recenter)", cv2.resize(vis, (640, 480)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                servo, locked_streak, t_start = SERVO_CENTER, 0, time.monotonic()
                if ser:
                    ser.write(f"V {SERVO_CENTER}\n".encode())
                wait_until = time.monotonic() + 40 / SERVO_DEG_PER_S + 0.15
            i += 1
    finally:
        cam.close()
        log.close()
        if ser:
            ser.write(b"S\n")
            ser.close()
        print(f"logged {i} frames to {run_dir}")


if __name__ == "__main__":
    main()
