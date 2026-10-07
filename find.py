"""find.py -- "find X" pan controller: search by sweeping the camera, then keep X centered (v2a).

Pure decision logic: box in, pan angle out. No camera, no model, no serial --
so it's unit-tested in test_find.py without any hardware.

    model box (or None) + pan angle the frame was taken at
        --> FindController.update()
        --> new pan angle (int degrees)   --> serve.py emits it on /drive/stream
                                          --> car_bridge.py sends "V <angle>" to the Pico

States:
    SEARCH  target not seen: step the pan SEARCH_STEP deg, bouncing between the
            limits (95..175). One model call per step -- an empty answer is fast
            (~250 ms, 1 token), so a full 80 deg sweep takes ~4-5 s.
    TRACK   target seen: proportional correction toward the box center.
            Lost only after LOST_AFTER consecutive misses, because a single
            empty answer is often the model flickering, not the object leaving.

"Look, then move": every decision uses a frame captured AFTER the servo stopped
(serve.py waits settle_time()). Acting on a mid-pan frame would apply the same
correction twice -> overshoot, then hunting left-right.
"""

SERVO_CENTER = 135              # camera straight ahead (measured 2026-10-06, matches SerialDrive.ino)
SERVO_MIN, SERVO_MAX = 95, 175  # +-40 deg usable range (firmware caps the right side at 175)
SERVO_DEG_PER_S = 60            # firmware slews gradually at ~60 deg/s

# Horizontal field of view of the OV3660 + stock lens: NOT measured yet, 60 is a
# placeholder. It converts "box center is 0.5 half-widths right of center" into
# degrees: delta = err * HFOV/2. Too small -> slow to center, too big -> overshoot.
HFOV_DEG = 60

# Correct only 70% of the measured error per step: HFOV is a guess and the servo
# isn't perfectly repeatable, so a full correction tends to overshoot. 70% closes
# 91% of the error in two steps and 97% in three, without oscillating.
GAIN = 0.7

# |err| below this counts as centered -> don't move. 0.10 of the 160 px half-width
# = 16 px, about how much PaliGemma's box edges jitter between identical frames.
DEADBAND = 0.10

SEARCH_STEP = 15  # deg per search step: < half the ~60 deg FOV, so consecutive views overlap
LOST_AFTER = 2    # consecutive empty answers before TRACK gives up and goes back to SEARCH

# After a pan command: slew time + this much before a frame counts as "settled".
# 7 fps = up to 143 ms between frames, plus JPEG encode/WiFi/decode latency.
SETTLE_EXTRA_S = 0.3


def clamp_pan(angle):
    return int(round(min(SERVO_MAX, max(SERVO_MIN, angle))))


def settle_time(old_pan, new_pan):
    """Seconds to wait after commanding old_pan -> new_pan before trusting a frame."""
    return abs(new_pan - old_pan) / SERVO_DEG_PER_S + SETTLE_EXTRA_S


class FindController:
    def __init__(self, pan_sign=1, frame_w=320):
        # pan_sign: +1 if INCREASING the servo angle turns the camera RIGHT.
        # Unverified on this car -- if the camera turns away from the target,
        # run with pan_sign=-1. Search doesn't care; tracking does.
        self.pan_sign = pan_sign
        self.frame_w = frame_w
        self.pan = SERVO_CENTER
        self.state = "SEARCH"
        self.sweep_dir = +1   # +1 = sweep toward increasing servo angle
        self.misses = 0
        self.err = None       # last box-center error in [-1, 1], None when not seen
        self.centered = False

    def reset(self):
        """New prompt: forget the old target, start searching from where we are."""
        self.state, self.misses, self.err, self.centered = "SEARCH", 0, None, False

    def update(self, box, pan_at_capture):
        """box: [x1,y1,x2,y2] in frame pixels, or None. pan_at_capture: servo angle
        when that frame was taken. Returns the new pan angle (int)."""
        if box is None:
            self.misses += 1
            self.err, self.centered = None, False
            if self.state == "TRACK" and self.misses < LOST_AFTER:
                return self.pan  # one empty answer: hold still and look again
            self.state = "SEARCH"
            nxt = self.pan + self.sweep_dir * SEARCH_STEP
            if not SERVO_MIN <= nxt <= SERVO_MAX:  # hit a limit: bounce back the other way
                self.sweep_dir = -self.sweep_dir
                nxt = self.pan + self.sweep_dir * SEARCH_STEP
            self.pan = clamp_pan(nxt)
            return self.pan

        self.misses = 0
        self.state = "TRACK"
        x1, _, x2, _ = box
        half = self.frame_w / 2
        # err in [-1, 1]: -1 = box center at the left edge, 0 = centered, +1 = right edge
        self.err = ((x1 + x2) / 2 - half) / half
        self.centered = abs(self.err) <= DEADBAND
        if not self.centered:
            # Correct relative to where the camera pointed WHEN THIS FRAME WAS TAKEN,
            # not where it points now -- the box describes that view.
            delta = self.pan_sign * GAIN * self.err * HFOV_DEG / 2
            self.pan = clamp_pan(pan_at_capture + delta)
        # If we lose it later, start searching on the side it was drifting toward.
        self.sweep_dir = 1 if self.pan_sign * self.err >= 0 else -1
        return self.pan
