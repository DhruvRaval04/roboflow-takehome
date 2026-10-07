"""camera.py -- read live frames from the ESP32-S3 camera's MJPEG stream (v2).

    ESP32 (CameraStation firmware) --WiFi, MJPEG over HTTP--> this module --> numpy frames

The ESP32 serves "multipart/x-mixed-replace": one endless HTTP response made
of back-to-back JPEGs. OpenCV's FFmpeg backend understands that format, so
cv2.VideoCapture(url) hands us one decoded BGR frame per JPEG.

LATENCY GOTCHA: VideoCapture buffers frames internally. If the detector runs
slower than the camera (~7 fps now), read() returns ever-OLDER frames and the
car reacts to the past. LatestFrame fixes this with a background thread that
reads continuously and keeps ONLY the newest frame -- the detector always sees
"now", and frames it was too slow for are simply dropped.

Run standalone to view the stream:   .venv\\Scripts\\python.exe camera.py
"""
import sys
import threading
import time

import cv2

# DHCP lease from the router (2026-10-06). If it changes after a router reboot,
# use the ESP32's mDNS name instead: http://rccam.local:81/stream
STREAM_URL = "http://192.168.0.103:81/stream"


class LatestFrame:
    """Background reader that always holds the most recent frame."""

    def __init__(self, url=STREAM_URL):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise SystemExit(f"Can't open {url} -- is the camera powered and on the router?")
        self.frame = None          # latest BGR frame, shape (240, 320, 3) at QVGA
        self.frame_id = 0          # increments per new frame; lets callers skip repeats
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:             # WiFi hiccup: back off briefly, keep trying
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self):
        """Return (frame_id, frame). frame is None until the first JPEG arrives."""
        with self.lock:
            return self.frame_id, self.frame

    def close(self):
        # Stop the reader thread BEFORE releasing the capture -- releasing it
        # mid-read() crashes the thread (seen in the first test run).
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


if __name__ == "__main__":
    cam = LatestFrame(sys.argv[1] if len(sys.argv) > 1 else STREAM_URL)
    last_id, t0, n = 0, time.monotonic(), 0
    while True:
        fid, frame = cam.read()
        if frame is not None and fid != last_id:
            last_id, n = fid, n + 1
            fps = n / (time.monotonic() - t0)
            cv2.putText(frame, f"{fps:.1f} fps", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow("rccam (q to quit)", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cam.close()
