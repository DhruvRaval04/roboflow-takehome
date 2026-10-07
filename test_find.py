"""test_find.py -- FindController behaviour without hardware.  Run: .venv\\Scripts\\python.exe -m pytest -q"""
from find import DEADBAND, SEARCH_STEP, SERVO_CENTER, SERVO_MAX, SERVO_MIN, FindController, clamp_pan, settle_time

W = 320


def box_centered_at(cx, half_w=20):
    return [cx - half_w, 50, cx + half_w, 150]


def test_search_sweeps_and_bounces_within_limits():
    c = FindController()
    seen = [c.update(None, c.pan) for _ in range(20)]
    assert all(SERVO_MIN <= p <= SERVO_MAX for p in seen)
    assert max(seen) >= SERVO_MAX - SEARCH_STEP and min(seen) <= SERVO_MIN + SEARCH_STEP  # covers the range
    assert all(abs(b - a) == SEARCH_STEP for a, b in zip(seen, seen[1:]))  # never jumps, always one step


def test_found_target_right_of_center_pans_right_by_gain_times_half_fov():
    c = FindController(pan_sign=+1)
    # box center at x=240 -> err = (240-160)/160 = +0.5 -> delta = 0.7*0.5*30 = +10.5 -> 135+10.5 -> 146 (rounded)
    assert c.update(box_centered_at(240), SERVO_CENTER) == 146
    assert c.state == "TRACK" and abs(c.err - 0.5) < 1e-9


def test_pan_sign_flips_direction():
    c = FindController(pan_sign=-1)
    assert c.update(box_centered_at(240), SERVO_CENTER) == 124  # 135 - 10.5 -> 124


def test_centered_target_does_not_move():
    c = FindController()
    inside = W / 2 + DEADBAND * W / 2 * 0.9  # just inside the deadband
    assert c.update(box_centered_at(inside), 150) == SERVO_CENTER  # pan unchanged (controller still at center)
    assert c.centered


def test_correction_is_relative_to_pan_at_capture():
    c = FindController()
    c.pan = 160  # commanded since, but the frame was taken at 140
    assert c.update(box_centered_at(240), 140) == 150  # 140 + 10.5 -> 150, not 160 + 10.5


def test_single_miss_holds_two_misses_resume_search_toward_last_side():
    c = FindController()
    c.update(box_centered_at(240), SERVO_CENTER)  # TRACK, target on the right
    held = c.pan
    assert c.update(None, held) == held and c.state == "TRACK"  # flicker: hold
    nxt = c.update(None, held)  # second miss: lost
    assert c.state == "SEARCH" and nxt == held + SEARCH_STEP  # searches right first, where it was heading


def test_reset_returns_to_search_keeping_position():
    c = FindController()
    c.update(box_centered_at(240), SERVO_CENTER)
    c.reset()
    assert c.state == "SEARCH" and c.pan == 146


def test_clamp_and_settle():
    assert clamp_pan(500) == SERVO_MAX and clamp_pan(-5) == SERVO_MIN
    assert abs(settle_time(135, 165) - (30 / 60 + 0.3)) < 1e-9
