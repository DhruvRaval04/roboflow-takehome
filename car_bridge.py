"""car_bridge.py -- the ONLY process that talks to the car: command stream in, USB serial out (v2a).

    serve.py  ──/drive/stream (10 JSON msgs/s, server-sent events)──►  car_bridge.py  ──COM7──►  Pico (SerialDrive)
    {"pan": 150, "wheels": [0,0,0,0], ...}                              "V 150\n"  (only when pan changes)
                                                                        "M 0 0 0 0\n" every 100 ms (heartbeat)

Why a separate process: Windows lets one process own a COM port. Keeping it
here means serve.py (and its 2.6 GB model) can restart without dropping the
car, and THIS small process is the single place that enforces safety:

  1. STALE STREAM -> STOP. If no message arrives for STALE_S (server crashed,
     hung, or restarting), send "S" every tick until messages return. The
     firmware's own 500 ms watchdog is the second layer.
  2. WHEELS LOCKED. v2a is pan-only: wheel values are forced to 0 unless you
     pass --allow-wheels. A controller bug can't drive the car off the table.
  3. PAN CLAMP. 95..175 deg here, even though the firmware accepts 10..175:
     outside 95..175 the camera hits the chassis (measured usable range).
  4. EXIT -> STOP. "S" on every exit path (q, Ctrl+C, exception).

Run (after serve.py is up):
    .venv\\Scripts\\python.exe car_bridge.py              # COM7
    .venv\\Scripts\\python.exe car_bridge.py --dry-run    # no serial: print what WOULD be sent
"""
import argparse
import json
import threading
import time
import urllib.request


from find import SERVO_CENTER, SERVO_MAX, SERVO_MIN  # noqa: E402

TICK_S = 0.1    # 10 Hz output: 5x margin under the firmware's 500 ms watchdog
STALE_S = 1.0   # no stream message for this long = server gone -> stop


class StreamReader:
    """Background thread: keeps the newest /drive/stream message + when it arrived."""

    def __init__(self, url):
        self.url = url
        self.msg, self.t_msg = None, 0.0
        self.connected = False
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                # timeout = max silence on the socket before we treat it as dead
                with urllib.request.urlopen(self.url, timeout=STALE_S * 2) as resp:
                    self.connected = True
                    for raw in resp:  # server-sent events: lines like b'data: {...}\n', blank line between
                        line = raw.decode("utf-8", errors="replace").strip()
                        if line.startswith("data: "):
                            msg = json.loads(line[6:])
                            with self.lock:
                                self.msg, self.t_msg = msg, time.monotonic()
            except Exception:  # server not up yet / restarted / socket timeout: retry
                pass
            self.connected = False
            time.sleep(1.0)

    def latest(self):
        with self.lock:
            age = time.monotonic() - self.t_msg if self.t_msg else float("inf")
            return self.msg, age


class Car:
    """USB serial to SerialDrive, or a printer in --dry-run."""

    def __init__(self, port, dry_run):
        self.dry_run = dry_run
        if dry_run:
            self.ser = None
        else:
            from drive_keys import open_car  # reuses the PONG handshake: proves COM7 is SerialDrive
            self.ser = open_car(port)

    def send(self, line):
        if self.dry_run:
            if not line.startswith("M 0 0 0 0"):  # heartbeat would flood the console
                print(f"  -> {line}")
            return
        self.ser.write((line + "\n").encode())

    def drain(self):
        """Print anything the firmware said that isn't a plain OK (ERR / WATCHDOG lines)."""
        if self.ser is None:
            return
        for line in self.ser.read(self.ser.in_waiting or 0).decode(errors="replace").splitlines():
            if line and not line.startswith("OK"):
                print(f"  car says: {line}")

    def close(self):
        if self.ser:
            self.ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--port", default="COM7")
    ap.add_argument("--dry-run", action="store_true", help="don't open serial; print commands instead")
    ap.add_argument("--allow-wheels", action="store_true", help="forward wheel values (v2b+). Default: forced 0")
    args = ap.parse_args()

    car = Car(args.port, args.dry_run)
    car.send("S")                      # known state before anything else: wheels stopped,
    car.send(f"V {SERVO_CENTER}")      # camera straight ahead
    stream = StreamReader(args.server + "/drive/stream")
    print(f"bridge up: {args.server}/drive/stream -> {'(dry run)' if args.dry_run else args.port}"
          f" | wheels {'ENABLED' if args.allow_wheels else 'locked to 0'}")

    last_pan, was_stale, last_status = SERVO_CENTER, True, 0.0
    try:
        while True:
            t0 = time.monotonic()
            msg, age = stream.latest()
            stale = msg is None or age > STALE_S
            if stale:
                car.send("S")  # every tick while stale -- cheap, and survives a missed line
                if not was_stale:
                    print(f"STREAM STALE ({age:.1f}s since last message) -> stop sent")
            else:
                if was_stale:
                    print("stream live")
                wheels = [int(w) for w in msg.get("wheels", [0, 0, 0, 0])] if args.allow_wheels else [0, 0, 0, 0]
                car.send("M " + " ".join(str(max(-100, min(100, w))) for w in wheels))  # heartbeat + wheels
                pan = msg.get("pan")
                if pan is not None:  # None = detect mode: leave the servo where it is
                    pan = int(min(SERVO_MAX, max(SERVO_MIN, pan)))
                    if pan != last_pan:  # the servo holds its angle by itself: send only changes
                        car.send(f"V {pan}")
                        last_pan = pan
            was_stale = stale
            car.drain()

            if t0 - last_status > 2.0 and not stale:  # compact status every 2 s
                print(f"  [{msg.get('state')}] pan {last_pan}  detect {msg.get('prompt')!r}  err {msg.get('err')}")
                last_status = t0
            time.sleep(max(0.0, TICK_S - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        car.send("S")  # stop instantly on exit; the firmware watchdog would anyway, 0.5 s later
        time.sleep(0.05)
        car.close()
        print("bridge stopped, car stopped.")


if __name__ == "__main__":
    main()
