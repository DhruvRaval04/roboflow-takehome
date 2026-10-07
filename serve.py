"""serve.py -- camera + PaliGemma 2 + "find X" controller, live in the browser (v2a).

  MAIN process (no torch, no model)                              MODEL process (spawned)
  ─────────────────────────────────                              ───────────────────────
  ESP32 MJPEG ──► LatestFrame ─┬─► control thread ──(frame, prompt)──► Grounder.locate()
  (192.168.0.103:81/stream)    │     │    ▲                              PaliGemma 2, 4-bit, RTX 4050
                               │     │    └────(boxes, raw, ms)─────────┘
                               │     └─► FindController (find.py): box -> pan angle   [--mode find]
                               │
                               └─► display thread: EVERY camera frame + latest boxes -> JPEG (~7 fps)

  browser     ◄── http://localhost:8000/detect   page (prompt box, video, live command stream)
              ◄── /detect/stream                 annotated MJPEG
              ◄── /detect/status                 JSON: prompt, model timing, controller state
  car_bridge  ◄── /drive/stream                  COMMAND STREAM: 10 JSON msgs/s (server-sent events)
     └──► COM7 ──► Pico: "V <pan>" + "M 0 0 0 0" heartbeat

Modes:  --mode detect   boxes only; command stream says pan=null (bridge leaves the servo alone)
        --mode find     search by sweeping the pan, then keep the target centered (pan only, no wheels)

This process never opens COM7 -- car_bridge.py is the only thing that talks to
the Pico (Windows allows one owner per COM port), so restarting this server
doesn't drop the car connection, and the bridge stops the car if this dies.

WHY TWO PROCESSES: with the model as a thread, one detection took ~1.76 s in
the server vs ~0.85 s standalone (measured 2026-10-06). PaliGemma generates its
answer one token at a time -- hundreds of tiny GPU calls, each needing Python's
interpreter lock (GIL) -- while the display/HTTP threads resize and JPEG-encode
7 frames/s in the same interpreter, fighting for that lock. A separate process
has its own interpreter and its own GIL, so neither side slows the other.
Only the MAIN process reads the camera: the ESP32 serves ONE stream client.

Box lag: the box is from a frame up to ~1 s old; the overlay prints its age.

NEXT (v2, follow mode): the same box results become pan/drive commands for the
Pico over USB serial. Only the bridge's consumer changes; this split stays.

The ESP32 can't host this (raw JPEGs only, no model), so the laptop serves it.
The ESP32 allows ONE stream client: close any other tab on 192.168.0.103:81.

Run:   .venv\\Scripts\\python.exe serve.py mug        then open http://localhost:8000/detect
"""
import argparse
import html
import json
import multiprocessing as mp
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import supervision as sv

from camera import LatestFrame  # noqa: E402  (newest-frame-only reader, see camera.py)

# NOTE: no `from vlm import Grounder` here. Importing it would load torch +
# CUDA into the main process for nothing; only model_worker() imports it.
from find import SERVO_CENTER, FindController, settle_time  # noqa: E402

COMMAND_HZ = 10  # /drive/stream rate: matches the Pico watchdog's needs (500 ms timeout, 5x margin)

# Shown at 2x (640x480): native 320x240 is too small to read labels on.
# Boxes are scaled by the same factor, since supervision draws in pixel coords.
SCALE = 2

# supervision's annotators take an image + sv.Detections and draw.
# BoxAnnotator = rectangle per detection; LabelAnnotator = filled text tag at the
# box's top-left, with the text we pass per detection (here: the exact prompt).
box_annotator = sv.BoxAnnotator(thickness=2)
# smart_position=True: the tag normally sits ABOVE the box's top-left corner, so
# a box touching the top edge (y1=0, common when an object fills the frame) puts
# the label off-image and it silently disappears. smart_position nudges it inside.
label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, text_padding=6, smart_position=True)


class State:
    """Shared between the model thread, the display thread and HTTP handlers."""

    def __init__(self, prompt, mode):
        self.lock = threading.Lock()
        self.prompt = prompt          # text conditioning: PaliGemma gets "detect <prompt>"
        self.mode = mode              # "detect" or "find"
        # controller output -- written by the control thread, read by /drive/stream + display
        self.pan = SERVO_CENTER       # commanded servo angle
        self.ctrl_state = "IDLE" if mode == "detect" else "SEARCH"
        self.ctrl_err = None          # last box-center error [-1, 1]
        # chat questions from the page -> control thread (which owns the model calls)
        self.chat_q = queue.Queue()
        # latest model answer -- written by the model thread, read by the display thread
        self.dets = sv.Detections.empty()
        self.dets_prompt = prompt     # prompt that PRODUCED self.dets (lags self.prompt after a change)
        self.dets_time = 0.0          # time.monotonic() of the frame the model looked at
        self.model_ms = 0.0
        self.raw = ""
        self.model_calls = 0
        # latest annotated JPEG -- written by the display thread, read by stream handlers
        self.jpeg = None
        self.jpeg_id = 0
        self.new_jpeg = threading.Condition(self.lock)


def model_worker(job_q, result_q, model, api_key):
    """Runs in the MODEL process. Loads PaliGemma once, then answers jobs forever.

    Two job kinds, same model:
      ("detect", frame, obj)       -> ("detect", xyxy Nx4 float in frame pixels, raw text, ms)
      ("ask",    frame, question)  -> ("ask", answer text, prompt actually sent, ms)
    frame = 240x320x3 uint8 BGR. Plain numpy/str only cross the process boundary
    (they're pickled); not sv.Detections, so the protocol doesn't depend on
    supervision internals.
    """
    from vlm import Grounder  # torch + CUDA load HERE, in this process only

    g = Grounder(model, api_key=api_key)
    result_q.put(("ready", None, None, 0.0))
    while True:
        kind, frame, text = job_q.get()
        if kind == "ask":
            r = g.ask(frame, text)
            result_q.put(("ask", r["answer"], r["prompt"], r["ms"]))
        else:
            r = g.locate(frame, text)
            result_q.put(("detect", r["dets"].xyxy, r["raw"], r["ms"]))


class FrameSaver:
    """--save-dir: keep raw frames the model saw, as fine-tuning data.

    Saves the UNANNOTATED frame (no boxes drawn -- those would leak into
    training images) at most once per SAVE_EVERY_S, plus one line in
    frames.jsonl with the pan angle, prompt and the raw model's answer. Those
    answers are the zero-shot baseline predictions on exactly these frames.
    """

    SAVE_EVERY_S = 1.0  # the model runs ~1-4x/s; consecutive frames are near-duplicates

    def __init__(self, save_dir):
        from pathlib import Path
        self.dir = Path(save_dir)
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        self.log = open(self.dir / "frames.jsonl", "a")
        self.last, self.n = 0.0, 0

    def maybe_save(self, frame, prompt, mode, pan, raw, xyxy):
        now = time.time()
        if now - self.last < self.SAVE_EVERY_S:
            return
        self.last = now
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(now * 1000) % 1000:03d}.jpg"
        cv2.imwrite(str(self.dir / "images" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.log.write(json.dumps({"image": name, "t": round(now, 3), "prompt": prompt, "mode": mode, "pan": pan,
                                   "raw": raw, "xyxy": np.asarray(xyxy).reshape(-1, 4).round(1).tolist()}) + "\n")
        self.log.flush()
        self.n += 1


def control_loop(state, cam, job_q, result_q, controller, saver=None):
    """MAIN process: settled frame -> model -> (find mode) controller -> new pan. Repeat.

    Lockstep (one job in flight): the model does ~1-4 calls/s, so queuing frames
    would only make them stale. "Look, then move": after a pan command we ignore
    every frame that ARRIVED before the servo finished moving (settle_until), so
    each decision is based on a view taken at a known, stationary angle.
    """
    last_id, settle_until, ctrl_prompt, ctrl_mode = 0, 0.0, None, None
    while True:
        fid, frame = cam.read()
        now = time.monotonic()
        if now < settle_until:
            last_id = fid  # mark frames that arrived mid-pan as already "seen" = never analysed
            time.sleep(0.005)
            continue
        if frame is None or fid == last_id:  # wait for a frame that arrived after settling
            time.sleep(0.005)
            continue
        last_id, t_frame = fid, now

        # CHAT: a pending question takes this step's model call instead of
        # detection -- one GPU model, one job at a time. Uses this exact
        # (settled) frame, i.e. "the current scene" as of the question.
        try:
            chat = state.chat_q.get_nowait()
        except queue.Empty:
            chat = None
        if chat is not None:
            job_q.put(("ask", frame.copy(), chat["q"]))
            _, answer, prompt_sent, ms = result_q.get()
            chat.update(answer=answer, prompt=prompt_sent, ms=ms)
            chat["done"].set()  # wakes the waiting HTTP handler
            continue

        with state.lock:
            # Read once for this whole step: the page toggle / prompt box can change
            # these at any moment, and one step must use one consistent set.
            prompt, pan_at_capture, mode = state.prompt, state.pan, state.mode
        if prompt != ctrl_prompt or mode != ctrl_mode:
            # New target, or toggled detect -> find: forget the old target and
            # search from wherever the camera points now.
            controller.reset()
            ctrl_prompt, ctrl_mode = prompt, mode

        frame = frame.copy()               # the camera thread owns the original buffer
        job_q.put(("detect", frame, prompt))
        _, xyxy, raw, ms = result_q.get()  # blocks while the model runs; releases the GIL
        dets = sv.Detections(xyxy=np.asarray(xyxy, dtype=float).reshape(-1, 4))
        if saver:
            saver.maybe_save(frame, prompt, mode, pan_at_capture, raw, xyxy)

        new_pan = pan_at_capture
        if mode == "find":
            # v1 policy carried over: no confidence scores, so the first box wins.
            box = dets.xyxy[0].tolist() if len(dets) else None
            new_pan = controller.update(box, pan_at_capture)
            if new_pan != pan_at_capture:
                settle_until = time.monotonic() + settle_time(pan_at_capture, new_pan)
        with state.lock:
            state.dets, state.dets_prompt, state.dets_time = dets, prompt, t_frame
            state.model_ms, state.raw = ms, raw
            state.model_calls += 1
            if state.mode != mode:
                pass  # toggled while the model was running: drop this step's control output
            elif mode == "find":
                state.pan, state.ctrl_state, state.ctrl_err = new_pan, controller.state, controller.err
                if controller.state == "TRACK" and controller.centered:
                    state.ctrl_state = "LOCKED"  # tracking AND within the deadband
            else:
                # detect mode: camera holds wherever it is (stream sends pan=null)
                state.ctrl_state, state.ctrl_err = "IDLE", None


def display_loop(state, cam):
    """EVERY new camera frame + latest detections -> annotated JPEG. ~camera fps."""
    last_id = 0
    while True:
        fid, frame = cam.read()
        if frame is None or fid == last_id:
            time.sleep(0.005)
            continue
        last_id = fid
        with state.lock:
            dets, dets_prompt, dets_time = state.dets, state.dets_prompt, state.dets_time
            prompt, model_ms, raw = state.prompt, state.model_ms, state.raw
            mode, pan, ctrl_state = state.mode, state.pan, state.ctrl_state

        vis = cv2.resize(frame, (frame.shape[1] * SCALE, frame.shape[0] * SCALE))
        # Only draw boxes that answer the CURRENT prompt: right after you switch
        # "wall" -> "bottle", the old wall box must not be shown labelled "bottle".
        if len(dets) and dets_prompt == prompt:
            # class_id 0 for every box: there's one class (the prompt); the
            # annotators use class_id to pick a color.
            scaled = sv.Detections(xyxy=dets.xyxy * SCALE, class_id=np.zeros(len(dets), dtype=int))
            vis = box_annotator.annotate(vis, scaled)
            vis = label_annotator.annotate(vis, scaled, labels=[f"detect {prompt}"] * len(scaled))
        age = time.monotonic() - dets_time if dets_time else 0.0
        found = len(dets) if dets_prompt == prompt else "..."
        line = f'"{prompt}": {found} found | model {model_ms:.0f} ms | box age {age:.1f}s'
        cv2.putText(vis, line, (8, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if mode == "find":
            color = {"SEARCH": (0, 165, 255), "TRACK": (0, 255, 255), "LOCKED": (0, 255, 0)}.get(ctrl_state, (255, 255, 255))
            cv2.putText(vis, f"{ctrl_state}  pan {pan}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # center crosshair: "LOCKED" means the box center is within the deadband around this line
            cx = vis.shape[1] // 2
            cv2.line(vis, (cx, 0), (cx, vis.shape[0]), (255, 255, 255), 1)

        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with state.new_jpeg:
                state.jpeg, state.jpeg_id = buf.tobytes(), state.jpeg_id + 1
                state.new_jpeg.notify_all()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>PaliGemma 2 live</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:16px}}
img{{max-width:100%;border:1px solid #444}} input{{font-size:16px;padding:6px}}
button{{font-size:16px;padding:6px 12px}} #st{{color:#9cf;font-family:monospace}}
.modes a{{display:inline-block;padding:8px 16px;margin-right:6px;border:1px solid #555;border-radius:6px;
  color:#aaa;text-decoration:none}} .modes a.on{{background:#2a6;color:#fff;border-color:#2a6}}
#banner{{font-size:20px;font-weight:600;margin:12px 0}}</style>
</head><body>
<h2>PaliGemma 2 via Roboflow inference_models</h2>
<div class="modes">mode:
  <a id="m-detect" href="/detect?mode=detect" class="{detect_on}">DETECT &mdash; camera fixed, boxes only</a>
  <a id="m-find" href="/detect?mode=find" class="{find_on}">FIND &mdash; sweep, then keep target centered</a>
</div>
<div id="banner">{banner}</div>
<form action="/detect" method="get">new prompt: <code>detect</code>
<input name="q" placeholder="{prompt}"> <button>set</button></form>
<p>current prompt (live from server): <b id="cur">{prompt}</b></p>
<p><img src="/detect/stream"></p>
<p id="st"></p>
<h3>Ask about the current scene</h3>
<form id="chatf"><input id="chatq" size="50" placeholder="what color is the bottle cap?  |  describe  |  ocr">
<button>ask</button></form>
<p style="color:#999">PaliGemma task types (click to fill):
  <a href="#" class="chip">describe</a> &middot; <a href="#" class="chip">describe en</a> &middot;
  <a href="#" class="chip">ocr</a> &middot; <a href="#" class="chip">what color is the bottle cap?</a> &middot;
  <a href="#" class="chip">how many bottles are there?</a> &middot; <a href="#" class="chip">detect bottle ; cup ; perfume</a> &middot;
  <a href="#" class="chip">segment bottle</a></p>
<div id="chatlog" style="font-family:monospace;white-space:pre-wrap;color:#ddd"></div>
<details open style="margin-top:16px;color:#bbb;max-width:820px"><summary><b>Cheat sheet</b></summary>
<p><b>Car modes</b> (buttons at the top) &mdash; what the car does continuously:</p>
<ul>
  <li><b>DETECT</b>: runs <code>detect &lt;prompt&gt;</code> over and over, draws the box, camera stays still.</li>
  <li><b>FIND</b>: same detection + the camera sweeps 95&ndash;175&deg; to search, then keeps the target centered
      (SEARCH &rarr; TRACK &rarr; LOCKED). Needs <code>car_bridge.py</code> running to actually move the servo.</li>
</ul>
<p><b>Chat</b> &mdash; one-off questions about the <i>current</i> frame. PaliGemma isn't a chat model; it knows fixed task prefixes:</p>
<table style="border-collapse:collapse" cellpadding="4">
  <tr><th align="left">type this</th><th align="left">sent to the model</th><th align="left">you get</th></tr>
  <tr><td><code>describe</code> / <code>what do you see</code></td><td><code>caption en</code></td><td>short caption</td></tr>
  <tr><td><code>cap en</code> / <code>caption en</code></td><td>as typed</td><td>short caption</td></tr>
  <tr><td><code>describe en</code></td><td>as typed</td><td>longer description (slower, ~2 s)</td></tr>
  <tr><td><code>ocr</code></td><td>as typed</td><td>text visible in the frame</td></tr>
  <tr><td>any question, e.g. <code>what color is the cap?</code></td><td><code>answer en &lt;question&gt;</code></td><td>short answer (~0.3 s)</td></tr>
  <tr><td><code>detect bottle ; cup</code></td><td>as typed</td><td>pixel boxes per object</td></tr>
  <tr><td><code>segment bottle</code></td><td>as typed</td><td>outline tokens (raw model mostly returns just a box)</td></tr>
  <tr><td><code>question en yellow</code></td><td>as typed</td><td>a question whose answer is "yellow"</td></tr>
</table>
<p>Raw (un-fine-tuned) model: yes/no &amp; color questions are decent; captions invent details; counting is vague;
   bottle vs perfume get confused. Swap languages by replacing <code>en</code> (e.g. <code>caption es</code>).</p>
</details>
<p>command stream (<code>/drive/stream</code>, what car_bridge.py forwards to the Pico):</p>
<pre id="cmd" style="color:#8f8">waiting...</pre>
<script>
// Same stream the car bridge consumes -- what you see here is exactly what the car gets.
new EventSource('/drive/stream').onmessage = (e) => {{ document.getElementById('cmd').textContent = e.data; }};
document.querySelectorAll('.chip').forEach(a => a.onclick = (e) => {{
  e.preventDefault(); const box = document.getElementById('chatq'); box.value = a.textContent; box.focus();
}});
// Chat: POST the question; the server answers from the frame current at that moment.
document.getElementById('chatf').onsubmit = async (ev) => {{
  ev.preventDefault();
  const box = document.getElementById('chatq'), log = document.getElementById('chatlog');
  const q = box.value.trim(); if (!q) return; box.value = '';
  const line = document.createElement('div'); line.textContent = `you: ${{q}}\n  ...`; log.prepend(line);
  try {{
    const r = await (await fetch('/chat', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
                                           body: JSON.stringify({{q}})}})).json();
    line.textContent = r.error ? `you: ${{q}}\n  error: ${{r.error}}`
      : `you: ${{q}}\n  model: ${{r.answer || '(empty)'}}   [sent "${{r.prompt}}", ${{r.ms}} ms]`;
  }} catch (e) {{ line.textContent = `you: ${{q}}\n  error: ${{e}}`; }}
}};
// Poll the server so "current prompt" is the truth, even if someone else changed it.
setInterval(async () => {{
  try {{
    const s = await (await fetch('/detect/status')).json();
    document.getElementById('cur').textContent = 'detect ' + s.prompt;
    // Mode banner + toggle highlight come from the SERVER, so they're never stale.
    document.getElementById('m-detect').className = s.mode === 'detect' ? 'on' : '';
    document.getElementById('m-find').className = s.mode === 'find' ? 'on' : '';
    document.getElementById('banner').textContent = s.mode === 'find'
      ? `FIND mode: ${{s.state}} for "${{s.prompt}}" (pan ${{s.pan}})`
      : `DETECT mode: camera fixed, showing boxes for "${{s.prompt}}"`;
    document.getElementById('st').textContent =
      `model: ${{s.model_ms.toFixed(0)}} ms/call, ${{s.model_calls}} calls | raw: ${{s.raw || '(empty)'}}`;
  }} catch (e) {{}}
}}, 500);
</script>
</body></html>"""


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/detect":
                params = parse_qs(url.query)
                q = params.get("q", [""])[0].strip()
                mode = params.get("mode", [""])[0]
                if q or mode in ("detect", "find"):
                    with state.lock:
                        if q:  # /detect?q=banana -> change what PaliGemma is asked to find
                            state.prompt = q
                        if mode in ("detect", "find"):  # /detect?mode=find -> toggle (control loop resets)
                            state.mode = mode
                            state.ctrl_state = "SEARCH" if mode == "find" else "IDLE"
                    self.send_response(303)  # redirect so a page refresh doesn't re-submit
                    self.send_header("Location", "/detect")
                    self.end_headers()
                    return
                with state.lock:
                    prompt, mode, cs, pan = state.prompt, state.mode, state.ctrl_state, state.pan
                banner = (f'FIND mode: {cs} for "{prompt}" (pan {pan})' if mode == "find"
                          else f'DETECT mode: camera fixed, showing boxes for "{prompt}"')
                page = PAGE.format(prompt=html.escape(prompt), banner=html.escape(banner),
                                   detect_on="on" if mode == "detect" else "", find_on="on" if mode == "find" else "")
                self._send(200, "text/html; charset=utf-8", page.encode())
            elif url.path == "/detect/status":
                with state.lock:
                    s = {"prompt": state.prompt, "model_ms": state.model_ms,
                         "model_calls": state.model_calls, "raw": state.raw,
                         "mode": state.mode, "state": state.ctrl_state, "pan": state.pan}
                self._send(200, "application/json", json.dumps(s).encode())
            elif url.path == "/drive/stream":
                # COMMAND STREAM, as server-sent events: one endless text/event-stream
                # response, one "data: {json}\n\n" message every 1/COMMAND_HZ s.
                # It's a *state* stream, not a list of one-off commands: every message
                # repeats the full desired state (pan + wheels), so a consumer that
                # connects late, or drops a message, is correct on the very next one.
                # In detect mode pan is null = "don't touch the servo".
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seq = 0
                try:
                    while True:
                        with state.lock:
                            msg = {"seq": seq, "mode": state.mode, "state": state.ctrl_state,
                                   "prompt": state.prompt,
                                   "pan": state.pan if state.mode == "find" else None,
                                   "wheels": [0, 0, 0, 0],  # v2a: pan only. v2b+ fills these.
                                   "err": None if state.ctrl_err is None else round(state.ctrl_err, 3)}
                        self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                        self.wfile.flush()
                        seq += 1
                        time.sleep(1 / COMMAND_HZ)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # consumer disconnected
            elif url.path == "/detect/stream":
                # MJPEG = the same format the ESP32 serves: one endless HTTP response,
                # each part a full JPEG; the browser swaps the <img> on every part.
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = 0
                try:
                    while True:
                        with state.new_jpeg:
                            state.new_jpeg.wait_for(lambda: state.jpeg_id != last, timeout=5)
                            if state.jpeg_id == last:
                                continue  # camera stalled; keep the connection open
                            jpeg, last = state.jpeg, state.jpeg_id
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # browser tab closed
            else:
                self.send_response(302)
                self.send_header("Location", "/detect")
                self.end_headers()

        def do_POST(self):
            if urlparse(self.path).path != "/chat":
                self._send(404, "text/plain", b"not found")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                q = json.loads(self.rfile.read(n) or b"{}").get("q", "").strip()
            except (ValueError, json.JSONDecodeError):
                q = ""
            if not q:
                self._send(400, "application/json", b'{"error": "empty question"}')
                return
            chat = {"q": q, "done": threading.Event()}
            state.chat_q.put(chat)
            # Wait for the control thread: at most one in-flight model call
            # (~1 s) + a servo settle + the answer itself (<=40 tokens, ~4 s).
            if not chat["done"].wait(timeout=20):
                self._send(504, "application/json", b'{"error": "model busy or camera stalled"}')
                return
            self._send(200, "application/json", json.dumps(
                {"q": q, "answer": chat["answer"], "prompt": chat["prompt"], "ms": round(chat["ms"])}).encode())

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # keep the console for model status, not per-request logs

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="mug", help='initial thing to detect, e.g. "mug"')
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="paligemma2-3b-pt-224")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--mode", choices=["detect", "find"], default="detect",
                    help="detect = boxes only; find = sweep the pan to search, then keep the target centered")
    ap.add_argument("--pan-sign", type=int, choices=[1, -1], default=1,
                    help="+1 if increasing servo angle turns the camera RIGHT; flip if it tracks AWAY")
    ap.add_argument("--save-dir", default=None,
                    help="save raw frames (<=1/s) + frames.jsonl here, as fine-tuning data")
    args = ap.parse_args()
    saver = FrameSaver(args.save_dir) if args.save_dir else None

    state = State(args.prompt, args.mode)
    controller = FindController(pan_sign=args.pan_sign)
    # Windows can only "spawn" child processes (fresh interpreter that re-imports
    # this file), which is why main() must stay behind `if __name__ == "__main__"`.
    job_q, result_q = mp.Queue(maxsize=1), mp.Queue(maxsize=1)
    worker = mp.Process(target=model_worker, args=(job_q, result_q, args.model, args.api_key), daemon=True)
    worker.start()
    print("model process started; loading PaliGemma 2 from Roboflow (~65 s)...")
    cam = LatestFrame()  # open the camera now so the video shows while the model loads
    threading.Thread(target=display_loop, args=(state, cam), daemon=True).start()

    tag, _, _, _ = result_q.get()  # block until the model process says it's loaded
    assert tag == "ready", tag
    print(f"model ready (pid {worker.pid})")
    threading.Thread(target=control_loop, args=(state, cam, job_q, result_q, controller, saver), daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"open http://localhost:{args.port}/detect   (Ctrl+C to stop)")
    try:
        last_calls, last_frames, t_last = 0, 0, time.monotonic()
        while True:  # once per second: model calls/s and display frames/s, so speed is measured, not guessed
            time.sleep(1)
            with state.lock:
                calls, frames = state.model_calls, state.jpeg_id
                prompt, ms, raw = state.prompt, state.model_ms, state.raw
                ctrl = f" | {state.ctrl_state} pan {state.pan}" if state.mode == "find" else ""
                ctrl += f" | saved {saver.n}" if saver else ""
            dt = time.monotonic() - t_last
            print(f"video {(frames - last_frames) / dt:4.1f} fps | model {(calls - last_calls) / dt:3.1f} calls/s, "
                  f"{ms:4.0f} ms{ctrl} | detect {prompt!r} -> {raw or '(empty)'}")
            last_calls, last_frames, t_last = calls, frames, time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        cam.close()
        worker.terminate()  # don't leave a 2.6 GB GPU process behind


if __name__ == "__main__":
    main()
