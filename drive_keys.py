"""drive_keys.py -- drive the car from the laptop keyboard over USB serial (v1).

    laptop (this script) --USB serial COM7--> Pico W SerialDrive firmware --> wheels

Keys (press once; the car KEEPS doing it until you press another key):
    w = forward     s = backward     a = spin left     d = spin right
    space = stop    + / - = speed up / down    q = quit (stops the car first)
    j / l = servo -5 / +5 deg    k = servo back to straight ahead (135)

Why "press once" instead of "hold": Windows key-repeat waits ~500 ms before
repeating a held key. The firmware's watchdog stops the car after 500 ms of
silence, so hold-to-drive would stutter. Instead this script re-sends the
current command 10x per second (HEARTBEAT_S) no matter what -- that's what
keeps the watchdog fed. Kill this script (Ctrl+C, crash, close the window) and
the heartbeat stops, so the car stops within 0.5 s by itself.

Run:  .venv\\Scripts\\python.exe drive_keys.py [COM7]
"""
import msvcrt  # Windows-only: non-blocking single-key reads, no extra install
import sys
import time

import serial  # pyserial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM7"
# Angle that points the camera straight ahead (servo arm is offset on its
# spline; measured 2026-10-06). Must match SERVO_CENTER in SerialDrive.ino.
SERVO_CENTER = 135
# Symmetric pan limit around center (deg). Right side is capped by the firmware's
# SERVO_MAX = 175 (= 135 + 40), so the left side is limited to match.
PAN = 40
HEARTBEAT_S = 0.1  # 10 Hz -- 5x margin under the firmware's 500 ms watchdog

# Wheel mixes in the firmware's m1..m4 order. Assumed layout, taken from the
# Freenove kit's own Motor_M_Move() calls (turn left = (-,-,+,+)):
#   m1 = front-left, m2 = rear-left, m3 = rear-right, m4 = front-right
# UNVERIFIED on this car: if 'w' drives backwards, flip the sign of FORWARD;
# if 'a' spins right, swap the TURN signs.
FORWARD = (1, 1, 1, 1)
TURN_LEFT = (-1, -1, 1, 1)  # left side backward, right side forward

MOVES = {
    "w": FORWARD,
    "s": tuple(-x for x in FORWARD),
    "a": TURN_LEFT,
    "d": tuple(-x for x in TURN_LEFT),
    " ": (0, 0, 0, 0),
}
NAMES = {"w": "FORWARD", "s": "BACKWARD", "a": "SPIN LEFT", "d": "SPIN RIGHT", " ": "STOPPED"}


def open_car(port):
    # Opening the port with DTR asserted is what makes arduino-pico's USB CDC
    # actually send/accept data (same reason our PowerShell readers set DtrEnable).
    ser = serial.Serial(port, 115200, timeout=0.05, dsrdtr=False)
    ser.dtr = True
    time.sleep(0.3)
    ser.reset_input_buffer()
    # Handshake: prove the thing on this port is SerialDrive, not e.g. the ESP32.
    for _ in range(10):
        ser.write(b"P\n")
        if b"PONG" in ser.read(64):
            return ser
        time.sleep(0.1)
    raise SystemExit(f"No PONG from {port} -- is SerialDrive flashed and the Pico plugged in?")


def main():
    ser = open_car(PORT)
    speed = 30  # 1..100 of the firmware's (already slow) cap
    servo = SERVO_CENTER  # degrees; firmware clamps to 10..175 and slews at ~60 deg/s
    key = " "
    print(__doc__.split("Run:")[0])
    last_send = 0.0
    try:
        while True:
            # ---- keyboard ----
            while msvcrt.kbhit():
                ch = msvcrt.getwch().lower()
                if ch == "q":
                    return
                if ch in MOVES:
                    key = ch
                elif ch in "+=":
                    speed = min(100, speed + 10)
                elif ch in "-_":
                    speed = max(10, speed - 10)
                elif ch in "jlk":
                    servo = {"j": max(SERVO_CENTER - PAN, servo - 5), "l": min(SERVO_CENTER + PAN, servo + 5), "k": SERVO_CENTER}[ch]
                    # Sent once per keypress (not in the heartbeat): the servo
                    # holds its angle by itself, nothing to keep alive.
                    ser.write(f"V {servo}\n".encode())

            # ---- heartbeat: (re)send the current command every HEARTBEAT_S ----
            now = time.monotonic()
            if now - last_send >= HEARTBEAT_S:
                wheels = [m * speed for m in MOVES[key]]
                ser.write(f"M {wheels[0]} {wheels[1]} {wheels[2]} {wheels[3]}\n".encode())
                last_send = now
                # Firmware answers every line; anything other than OK is worth seeing.
                for line in ser.read(256).decode(errors="replace").splitlines():
                    if line and not line.startswith("OK"):
                        print(f"\n  car says: {line}")
                print(f"\r  {NAMES[key]:<10}  speed {speed:>3}  servo {servo:>3}   ", end="", flush=True)
            time.sleep(0.01)
    finally:
        # Explicit stop on every exit path (q, Ctrl+C, exception). The watchdog
        # would catch it anyway, but this stops the car instantly, not in 0.5 s.
        ser.write(b"S\n")
        time.sleep(0.05)
        ser.close()
        print("\nstopped.")


if __name__ == "__main__":
    main()
