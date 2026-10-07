"""vlm.py -- open-vocabulary object grounding with PaliGemma 2, all on Roboflow's stack (v1).

    320x240 BGR frame + "mug"
        --> inference_models.AutoModel("paligemma2-3b-pt-224")   weights from Roboflow's model registry,
                                                                  4-bit on the RTX 4050
        --> "<loc0187><loc0412><loc0590><loc0811> mug"            raw PaliGemma answer (~6 tokens)
        --> supervision.Detections.from_vlm(PALIGEMMA)            Roboflow's parser -> box in frame pixels

Every model in the results table loads through the SAME AutoModel call, only the ID changes:
    zero-shot : "paligemma2-3b-pt-224"          (Roboflow-hosted pretrained, no API key needed)
    SFT       : "<project>/<version>"          (our LoRA, uploaded with version.deploy, needs api_key)
    GRPO      : "<project>/<version>"          (same)

Standalone benchmark (one frame, prints box + latency, saves annotated jpg):
    .venv\\Scripts\\python.exe vlm.py mug --capture        # grab a live frame from the ESP32
    .venv\\Scripts\\python.exe vlm.py mug path\\to\\img.jpg  # or any image file
"""
import argparse
import time

import cv2
import numpy as np
import supervision as sv
import torch
from inference_models import AutoModel
from transformers import BitsAndBytesConfig

# Roboflow's alias for Google's PaliGemma 2, 3B params, 224x224 input, "pt" =
# pretrained-for-transfer (not instruction-tuned). Its pretraining mix includes
# detection, so it answers "detect <thing>" zero-shot -- and "pt" is the
# checkpoint Google intends for fine-tuning, which is what v2/v3 do.
MODEL_ID = "paligemma2-3b-pt-224"

# PaliGemma's native task-prefix syntax. Not a chat prompt: the model was
# trained on literal prefixes like "caption en", "detect cat ; dog", "segment cat".
# The processor wraps it as [256 image tokens][BOS]"detect mug\n" by itself.
PROMPT = "detect {obj}"

# LATENCY = ~200 ms fixed (vision encoder + reading the prompt) + ~80-130 ms PER
# GENERATED TOKEN on the 4050 (each token = one full pass through the 4-bit
# Gemma 2 LM). A full detection answer is 4 <loc> tokens + label + <eos> = 6+
# tokens, so it cost ~3x an empty answer (1 token: <eos>).
# The label just echoes our own prompt -- we already know it. So stop after 4:
#   nothing found -> 1st token is <eos>, generation ends (unchanged, 1 token)
#   found         -> exactly the 4 <loc> tokens; we re-attach the label ourselves
# Cost: only the FIRST box comes back (no "bottle ; bottle"); the find controller
# uses only the first box anyway.
MAX_NEW_TOKENS = 4
FULL_ANSWER_TOKENS = 20  # old behaviour (label + multi-object), kept for comparison/debug


class Grounder:
    def __init__(self, model=MODEL_ID, api_key=None, vision_fp16=True, full_answer=False):
        self.max_new_tokens = FULL_ANSWER_TOKENS if full_answer else MAX_NEW_TOKENS
        # AutoModel resolves the ID against Roboflow's registry, downloads + caches
        # the weights package, and picks the right class (PaliGemmaHF). On CUDA it
        # applies 4-bit NF4 quantization by default (inference_models/models/
        # paligemma/paligemma_hf.py): ~3B params * 0.5 byte ~= 2-3 GB, which is
        # what makes it fit next to Windows' display on the 4050's 6 GB.
        # api_key is only needed for our own fine-tuned "project/version" models.
        #
        # We override Roboflow's default 4-bit config for ONE reason: its default
        # also quantizes the SigLIP vision encoder, whose MLP width (4304) isn't a
        # multiple of bitsandbytes' 64-wide fast kernel -> the log warned "falling
        # back to slower implementation" on every frame. Keeping the vision
        # encoder + projector in fp16 (~400M params * 2 B = +0.8 GB) avoids that
        # slow path AND keeps full-precision image features -- box coordinates
        # come from those features, and our 320x240 frames have little to spare.
        # The 2.6B-param Gemma 2 language model stays 4-bit (it's what needs shrinking).
        # AutoModel forwards extra kwargs to PaliGemmaHF.from_pretrained.
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"] if vision_fp16 else None,
        )
        # Hard CUDA requirement. On CPU this 3B model takes tens of seconds per
        # frame -- and a CPU-only torch install (pip once silently swapped ours
        # for one) would otherwise just run slowly with no error at all.
        if not torch.cuda.is_available():
            raise SystemExit(f"CUDA not available (torch {torch.__version__}) -- refusing to run the VLM on CPU. "
                             "Reinstall torch from https://download.pytorch.org/whl/cu128")
        kwargs = {"api_key": api_key} if api_key else {}
        self.model = AutoModel.from_pretrained(model, quantization_config=quant,
                                               device=torch.device("cuda:0"), **kwargs)
        # Verify where the weights ACTUALLY landed (not where we asked): count
        # parameters per device. Every one must be on cuda:0.
        devices = {}
        for p in self.model._model.parameters():
            devices[str(p.device)] = devices.get(str(p.device), 0) + p.numel()
        print(f"[vlm] {torch.cuda.get_device_name(0)} | params per device: "
              + ", ".join(f"{d}={n / 1e6:.0f}M" for d, n in devices.items())
              + f" | VRAM allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        if any(not d.startswith("cuda") for d in devices):
            raise SystemExit(f"model weights not all on GPU: {devices}")

    def locate(self, frame_bgr, obj):
        """Return dict(box=[x1,y1,x2,y2] in frame pixels or None, raw=str, ms=float)."""
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        raw = self.model.prompt(
            images=frame_bgr,
            prompt=PROMPT.format(obj=obj),
            input_color_format="bgr",       # OpenCV frames; wrong order swaps red/blue ("red mug" breaks)
            max_new_tokens=self.max_new_tokens,
            do_sample=False,                # greedy: same frame -> same answer, required for a fair eval
            # Roboflow's default (True) can strip PaliGemma's <locXXXX> tokens --
            # they ARE the box. Keep them; supervision ignores <eos>/<pad> text.
            skip_special_tokens=False,
        )[0]
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000  # full call: preprocess + prefill + decode + decode-to-text

        raw = raw.replace("<eos>", "").replace("<pad>", "").strip()
        # Truncated answer = bare "<locA><locB><locC><locD>". supervision's
        # PaliGemma parser REQUIRES a label after the 4 locs -- without one it
        # silently returns 0 detections (verified). Re-attach our own prompt.
        if raw.endswith(">") and raw.count("<loc") == 4:
            raw = f"{raw} {obj}"
        # COORDINATES: each <locNNNN> is a bin in 0..1023 over the image's height
        # or width, in order (y_min, x_min, y_max, x_max). Because they are
        # NORMALIZED, the 320x240 -> 224x224 stretch inside the processor does
        # not matter: x = N/1024 * 320, y = N/1024 * 240. supervision does exactly
        # that given resolution_wh. (Verified: <loc0256><loc0128><loc0768><loc0640>
        # -> [40, 60, 200, 180] on 320x240.)
        dets = sv.Detections.from_vlm(sv.VLM.PALIGEMMA, raw, resolution_wh=(w, h), classes=[obj])
        # v1 policy: PaliGemma has no confidence scores, so if it returns several
        # boxes ("mug ; mug") take the first one it emitted.
        box = [float(v) for v in dets.xyxy[0]] if len(dets) else None
        # "dets" = the full sv.Detections (all boxes + class names), for supervision's annotators
        return {"box": box, "raw": raw, "ms": ms, "n_dets": len(dets), "dets": dets}

    def ask(self, frame_bgr, text):
        """Chat: free-form question about this frame -> dict(answer, prompt actually sent, ms)."""
        return _ask(self.model, frame_bgr, text)


# PaliGemma isn't a chat model: it was pretrained on literal task prefixes.
# Text starting with one of these goes to the model untouched; anything else is
# treated as a question and wrapped as "answer en <question>" (its VQA prefix).
# Full PaliGemma task list (model card): "cap/caption/describe <lang>", "ocr",
# "answer <lang> <q>", "question <lang> <answer>", "detect a ; b", "segment a".
TASK_PREFIXES = ("cap", "caption", "describe", "ocr", "answer", "question", "detect", "segment")
DESCRIBE_ALIASES = {"describe", "describe the scene", "what do you see", "what do you see?", "caption"}


def to_paligemma_prompt(text):
    t = " ".join(text.strip().split())
    if t.lower() in DESCRIBE_ALIASES:
        return "caption en"            # scene description
    if t.lower().split(" ", 1)[0] in TASK_PREFIXES:
        return t                        # already PaliGemma syntax, e.g. "ocr", "caption en"
    return f"answer en {t}"            # visual question answering


def _ask(model, frame_bgr, text, max_new_tokens=40):
    prompt = to_paligemma_prompt(text)
    h, w = frame_bgr.shape[:2]
    t0 = time.perf_counter()
    answer = model.prompt(
        images=frame_bgr, prompt=prompt, input_color_format="bgr",
        # Answers are short ("yellow", "2", "a bottle on a desk"); 40 tokens caps a
        # rambling base model at ~4 s instead of Roboflow's 400-token default (~40 s).
        max_new_tokens=max_new_tokens, do_sample=False,
        # Keep special tokens: for "detect"/"segment" the <loc>/<seg> tokens ARE
        # the answer. Strip only the end/padding markers ourselves.
        skip_special_tokens=False,
    )[0].replace("<eos>", "").replace("<pad>", "").strip()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    if prompt.startswith("detect "):
        # Turn "<loc..>x4 cup ; <loc..>x4 bottle" into readable pixel boxes
        # (same supervision parser as locate(); bins/1024 * frame size).
        classes = [c.strip() for c in prompt[len("detect "):].split(";") if c.strip()]
        dets = sv.Detections.from_vlm(sv.VLM.PALIGEMMA, answer, resolution_wh=(w, h), classes=classes)
        names = dets.data.get("class_name", [""] * len(dets))
        if len(dets):
            answer = "; ".join(f"{n} at [{', '.join(str(int(v)) for v in b)}]" for n, b in zip(names, dets.xyxy))
        else:
            answer = answer or "(nothing found)"
    return {"answer": answer, "prompt": prompt, "ms": ms}



def draw(frame, box, label):
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", help='what to find, e.g. "mug"')
    ap.add_argument("image", nargs="?", default=None, help="image path (omit with --capture)")
    ap.add_argument("--capture", action="store_true", help="grab a live frame from the ESP32 instead of a file")
    ap.add_argument("--model", default=MODEL_ID, help="Roboflow model ID or project/version")
    ap.add_argument("--api-key", default=None, help="Roboflow API key (only for project/version models)")
    ap.add_argument("--runs", type=int, default=5, help="timed runs after warmup")
    ap.add_argument("--full-answer", action="store_true",
                    help="let the model write label + extra boxes (old, slower) instead of stopping at 4 tokens")
    ap.add_argument("--quant-vision", action="store_true",
                    help="also 4-bit the vision encoder (Roboflow's default) -- for speed/accuracy comparison")
    args = ap.parse_args()

    if args.capture:
        import urllib.request
        # /capture returns one fresh JPEG -- simpler than opening the MJPEG stream for one frame
        jpg = urllib.request.urlopen("http://192.168.0.103/capture", timeout=5).read()
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("capture.jpg", frame)  # keep it so the same frame can be re-tested later
        print(f"captured {frame.shape[1]}x{frame.shape[0]} -> capture.jpg")
    else:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"can't read {args.image}")

    t0 = time.perf_counter()
    g = Grounder(args.model, api_key=args.api_key, vision_fp16=not args.quant_vision, full_answer=args.full_answer)
    print(f"model loaded in {time.perf_counter() - t0:.1f} s")

    g.locate(frame, args.obj)  # warmup: first call pays CUDA kernel setup/allocation, don't time it
    results = [g.locate(frame, args.obj) for _ in range(args.runs)]
    ms = sorted(r["ms"] for r in results)
    r = results[-1]
    print(f"box={r['box']}  n_dets={r['n_dets']}")
    print(f"raw: {r['raw']!r}")
    print(f"latency over {args.runs} runs: median {ms[len(ms) // 2]:.0f} ms  (min {ms[0]:.0f}, max {ms[-1]:.0f})")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    cv2.imwrite("vlm_out.jpg", draw(frame, r["box"], args.obj))
    print("wrote vlm_out.jpg")
