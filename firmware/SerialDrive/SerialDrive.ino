// SerialDrive -- the Pico W takes wheel commands from the laptop over USB serial.
//
//   laptop Python --USB serial (COM7)--> this sketch --PWM--> H-bridges --> 4 motors
//
// PROTOCOL (one text line per command, '\n' terminated):
//   M a b c d   set the 4 wheel speeds, each -100..100 (sign = direction).
//               Order is the Freenove kit's Motor_M_Move(m1, m2, m3, m4) order.
//   S           stop all wheels now
//   V a         point the servo at angle a (degrees, clamped 10..175; SERVO_CENTER = straight ahead)
//   P           ping -> replies "PONG" (lets Python confirm it found the car)
// Every valid command is answered "OK ..." and every bad line "ERR ...", so the
// laptop always knows whether the car understood it.
//
// SAFETY (everything on the car is barely bolted on, so slow is mandatory):
//   1. Speed cap lives HERE, not in Python: 100 from the laptop = MAX_DUTY
//      (~39% of battery voltage). A buggy script can't make the car go faster.
//   2. Watchdog: if no command arrives for WATCHDOG_MS, stop all wheels. If the
//      Python script crashes or the cable is yanked, the car stops by itself
//      instead of running the last command forever. Python must therefore keep
//      re-sending the current command (~10x/s) even when nothing changes.

// ---- Wheel wiring (verified 2026-10-06 with MotorWireTest: all 4 spin both ways) ----
// Each motor = an H-bridge with two inputs. PWM on `fwd` and 0 on `rev` spins it
// the kit's "forward" way; swap the two to reverse. Both 0 = coast.
struct Wheel {
  uint8_t fwd, rev;          // GP numbers (pins from Freenove_4WD_Car_For_Pico_W.h)
  int target;                // last commanded speed -100..100
  unsigned long kickUntil;   // millis() when this wheel's start-up kick ends (0 = none)
};
Wheel wheels[4] = {
  {18, 19, 0, 0},  // m1  shield connector M1 [19][18]
  {21, 20, 0, 0},  // m2  shield connector    [21][20]
  { 7,  6, 0, 0},  // m3  shield connector    [6][7]
  { 9,  8, 0, 0},  // m4  shield connector M4 [9][8]
};

// ---- Speed tuning (measured on this car, analogWrite range 0..255) ----
// Duty = fraction of the ~8 V battery the motor sees on average.
//   51 (20%): stall -- motors whine, don't turn.
//   90 (35%): turns slowly once moving; one stiff gearbox needs a kick to start.
const int MIN_DUTY = 70;    // speed 1   -> 27%: slowest that still turns (with kick)
const int MAX_DUTY = 100;   // speed 100 -> 39%: hard ceiling, deliberately slow
// Kick-start: static friction (gearbox at rest) > kinetic friction (turning).
// A short burst breaks the wheel loose, then it settles to the slow duty.
// Kept short so the car only twitches -- it doesn't lurch.
const int KICK_DUTY = 180;  // ~70% for...
const unsigned long KICK_MS = 70;

// ---- Servo (shield header [13] -> jumper -> Pico GP13; PIN_SERVO1 in the kit .h) ----
// A hobby servo reads a 50 Hz pulse train: 0.5 ms pulse = 0 deg, 2.5 ms = 180 deg.
// The Servo library generates that in hardware PWM; we only pick the angle.
// The servo itself snaps to a new angle as fast as it can (~0.1 s / 60 deg),
// which would yank whatever is mounted on it -- so we SLEW: step the commanded
// angle 1 deg at a time every SERVO_STEP_MS (~60 deg/s) toward the target.
#include <Servo.h>
const uint8_t SERVO_PIN = 13;
// Widened from the kit's cautious 30..150: with the arm offset (center = 135),
// 150 left only 15 deg of turn on one side. The servo itself spans 0..180, but
// many SG90s hit their internal end stop at the extremes and buzz/strain
// there -- so stop 5 deg short. Pan range around center: 135 +/- 40.
const int SERVO_MIN = 10, SERVO_MAX = 175;
const unsigned long SERVO_STEP_MS = 16;     // 1 deg per 16 ms ~= 60 deg/s
Servo servo;
// The servo arm sits a few teeth off on its spline, so the angle that points
// the camera STRAIGHT AHEAD is not 90 on this car -- measured with j/l keys.
const int SERVO_CENTER = 135;
int servoNow = SERVO_CENTER, servoTarget = SERVO_CENTER;
unsigned long servoLastStep = 0;

const unsigned long WATCHDOG_MS = 500;
unsigned long lastCmdMs = 0;
bool stoppedByWatchdog = true;  // start "stopped"; first command clears it

// Map a speed 1..100 onto MIN_DUTY..MAX_DUTY. (Linear 0..MAX would put speeds
// 1-70 into the stall zone, where the wheel whines but doesn't move.)
int speedToDuty(int s) {
  if (s <= 0) return 0;
  return MIN_DUTY + (MAX_DUTY - MIN_DUTY) * (s - 1) / 99;
}

// Push one wheel's state to its two H-bridge pins.
void applyWheel(Wheel& w, unsigned long now) {
  int duty = speedToDuty(abs(w.target));
  if (duty > 0 && now < w.kickUntil) duty = KICK_DUTY;  // still inside the kick window
  if (w.target > 0)      { analogWrite(w.rev, 0); analogWrite(w.fwd, duty); }
  else if (w.target < 0) { analogWrite(w.fwd, 0); analogWrite(w.rev, duty); }
  else                   { analogWrite(w.fwd, 0); analogWrite(w.rev, 0); }  // coast
}

void setWheel(int i, int speed, unsigned long now) {
  speed = constrain(speed, -100, 100);
  Wheel& w = wheels[i];
  // Kick only when the wheel is starting from rest or reversing -- i.e. when
  // static friction (or the motor's own momentum) has to be overcome. Changing
  // speed in the same direction needs no kick.
  bool starting = (speed != 0) && (w.target == 0 || (w.target > 0) != (speed > 0));
  if (starting) w.kickUntil = now + KICK_MS;
  w.target = speed;
  applyWheel(w, now);
}

void stopAll() {
  for (auto& w : wheels) { w.target = 0; w.kickUntil = 0; applyWheel(w, 0); }
}

void handleLine(char* line, unsigned long now) {
  char cmd = line[0];
  if (cmd == 'M') {
    int v[4];
    // sscanf returns how many numbers it parsed; anything but 4 = malformed.
    if (sscanf(line + 1, "%d %d %d %d", &v[0], &v[1], &v[2], &v[3]) != 4) {
      Serial.println("ERR need: M a b c d");
      return;
    }
    for (int i = 0; i < 4; i++) setWheel(i, v[i], now);
    lastCmdMs = now;
    stoppedByWatchdog = false;
    Serial.printf("OK M %d %d %d %d\n", wheels[0].target, wheels[1].target,
                  wheels[2].target, wheels[3].target);
  } else if (cmd == 'S') {
    stopAll();
    lastCmdMs = now;
    Serial.println("OK S");
  } else if (cmd == 'V') {
    int a;
    if (sscanf(line + 1, "%d", &a) != 1) { Serial.println("ERR need: V angle"); return; }
    servoTarget = constrain(a, SERVO_MIN, SERVO_MAX);
    // Note: V does NOT feed the drive watchdog -- pointing the servo is not
    // "the laptop is driving", so it must not keep the wheels alive.
    Serial.printf("OK V %d\n", servoTarget);
  } else if (cmd == 'P') {
    lastCmdMs = now;  // a ping also counts as "laptop is alive"
    Serial.println("PONG");
  } else if (cmd != '\0') {
    Serial.printf("ERR unknown '%c'\n", cmd);
  }
}

void setup() {
  Serial.begin(115200);  // USB CDC: baud is ignored, but keep it consistent with Python
  stopAll();             // known state before anything else: all H-bridge inputs low
  pinMode(LED_BUILTIN, OUTPUT);
  // 500/2500 us = the pulse widths for 0/180 deg (same as the kit's map()).
  servo.attach(SERVO_PIN, 500, 2500);
  servo.write(servoNow);  // straight ahead on boot
  Serial.println("READY SerialDrive");
}

void loop() {
  unsigned long now = millis();

  // ---- Read serial without blocking: collect chars until '\n' ----
  // (readStringUntil() would block the loop and freeze the watchdog/kicks.)
  static char buf[64];
  static uint8_t len = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;           // tolerate Windows "\r\n" line endings
    if (c == '\n') { buf[len] = '\0'; handleLine(buf, now); len = 0; }
    else if (len < sizeof(buf) - 1) buf[len++] = c;
    // overlong lines are truncated, then rejected by handleLine's parsing
  }

  // ---- End kick windows: drop each kicked wheel to its slow cruising duty ----
  for (auto& w : wheels) {
    if (w.kickUntil && now >= w.kickUntil) { w.kickUntil = 0; applyWheel(w, now); }
  }

  // ---- Servo slew: one degree per SERVO_STEP_MS toward the target ----
  if (servoNow != servoTarget && now - servoLastStep >= SERVO_STEP_MS) {
    servoNow += (servoTarget > servoNow) ? 1 : -1;
    servo.write(servoNow);
    servoLastStep = now;
  }

  // ---- Watchdog: silence from the laptop = stop ----
  if (!stoppedByWatchdog && now - lastCmdMs > WATCHDOG_MS) {
    stopAll();
    stoppedByWatchdog = true;
    Serial.println("WATCHDOG stop");
  }

  // LED on = wheels are being driven; off = stopped. Visible proof from across
  // the table of whether the car *thinks* it's moving.
  // Only write on change: on the Pico W the LED hangs off the CYW43 WiFi chip,
  // so each digitalWrite is an SPI transaction, not a cheap register poke.
  static bool ledState = false;
  bool moving = false;
  for (auto& w : wheels) moving |= (w.target != 0);
  if (moving != ledState) { ledState = moving; digitalWrite(LED_BUILTIN, moving); }
}
