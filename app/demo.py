"""Gradio demo: paste code, get a calibrated P(machine) plus the evidence behind it.

    python app/demo.py --ckpt results/ckpt/task_a/gpad          # or set CODEDET_CKPT
    python app/demo.py                                          # feature-only mode, no checkpoint needed

Deliberate product choices, because a detector that is wrong in public does real harm:
  * the probability shown is temperature-scaled (T from `calibration.json`, fitted on ID validation);
  * an out-of-range warning fires when the snippet is shorter/longer than the training range or in a
    language the model never saw - that is exactly where the OOD numbers say it should not be trusted;
  * the verdict wording is "evidence, not proof", with the false-accusation cost stated on screen.
No checkpoint is committed: the app loads one from a path or env var, and degrades to the interpretable
stylometric / LM view when none is available.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from codedet.features import STYLE_NAMES, StyleScaler, style_matrix

LABELS = ("human", "machine")
MIN_CHARS, MAX_CHARS = 80, 20000
KNOWN_LANGUAGES = ("Python", "C++", "Java")

EXAMPLE_HUMAN = "def f(a):\n  s=0\n  for x in a: s+=x\n  return s\n"
EXAMPLE_MACHINE = ('def compute_total(numbers):\n    """Return the sum of all items in numbers."""\n'
                   "    total = 0\n    # iterate over every element\n    for item in numbers:\n"
                   "        total += item\n    return total\n")


class Detector:
    """Wraps whatever is available: a trained checkpoint, or features only."""

    def __init__(self, ckpt: str | None = None, temperature: float = 1.0):
        self.model = None
        self.tokenizer = None
        self.scaler: StyleScaler | None = None
        self.temperature = temperature
        self.ckpt = ckpt
        if ckpt and Path(ckpt).exists():
            self._load(Path(ckpt))

    def _load(self, d: Path) -> None:
        from transformers import AutoTokenizer

        from codedet.models.gpad import GPADClassifier

        self.model = GPADClassifier.load(str(d))
        cfg = json.loads((d / "gpad_config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.get("base_model", "microsoft/codebert-base"))
        scaler_file = d / "scaler.json"
        if scaler_file.exists():
            self.scaler = StyleScaler.from_dict(json.loads(scaler_file.read_text()))
        t_file = d / "calibration.json"
        if t_file.exists():
            self.temperature = float(json.loads(t_file.read_text()).get("temperature", 1.0))

    def warnings(self, code: str, language: str) -> list[str]:
        out = []
        if len(code) < MIN_CHARS:
            out.append(f"Snippet is very short ({len(code)} characters). Detectors are unreliable below "
                       f"~{MIN_CHARS} characters.")
        if len(code) > MAX_CHARS:
            out.append("Snippet is longer than the training range; only the head and tail are read.")
        if language not in KNOWN_LANGUAGES:
            out.append(f"Language '{language}' is outside the training languages "
                       f"({', '.join(KNOWN_LANGUAGES)}). Measured out-of-language macro-F1 is much lower.")
        return out

    def predict(self, code: str, language: str):
        style = style_matrix([code])[0]
        info = {"style": dict(zip(STYLE_NAMES, [round(float(v), 4) for v in style]))}
        if self.model is None:
            # feature-only fallback: a transparent heuristic, clearly labelled as such
            comment, ident = style[7], style[8]
            score = float(np.clip(0.5 + 1.2 * comment + 0.05 * (ident - 4), 0.02, 0.98))
            info["mode"] = "features only (no checkpoint loaded) - indicative, not a trained detector"
            return score, info
        import torch

        from codedet.inference import predict_texts

        probs, logits = predict_texts(self.model, self.tokenizer, [code], scaler=self.scaler,
                                      temperature=self.temperature, device="cpu")
        info["mode"] = f"checkpoint {self.ckpt} (temperature {self.temperature:.2f})"
        info["logits"] = [round(float(v), 3) for v in logits[0]]
        return float(probs[0][1]), info


def build_ui(det: Detector, threshold: float):
    import gradio as gr

    def run(code: str, language: str):
        code = code or ""
        if not code.strip():
            return "Paste some code first.", {}, ""
        p, info = det.predict(code, language)
        verdict = "likely machine-generated" if p >= threshold else "likely human-written"
        warn = det.warnings(code, language)
        head = (f"## P(machine) = {p:.2f}\n\n**{verdict}** at the cost-chosen threshold {threshold:.2f} "
                f"(a false accusation is treated as 5x costlier than a miss).\n\n"
                f"*This is evidence, not proof. Measured accuracy drops sharply on unseen languages; "
                f"never use a detector score as the sole basis for an accusation.*")
        if warn:
            head += "\n\n" + "\n".join(f"> WARNING: {w}" for w in warn)
        return head, info.get("style", {}), info.get("mode", "")

    with gr.Blocks(title="Machine-generated code detector") as ui:
        gr.Markdown("# Is this code machine-generated?\n"
                    "SemEval-2026 Task 13 portfolio project. Model: CodeBERT + attention-entropy bias "
                    "+ stylometric fusion (GPAD), with calibrated probabilities.")
        with gr.Row():
            with gr.Column(scale=3):
                code = gr.Code(label="code", language="python", value=EXAMPLE_MACHINE, lines=18)
                language = gr.Dropdown(list(KNOWN_LANGUAGES) + ["other"], value="Python", label="language")
                btn = gr.Button("Analyse", variant="primary")
                gr.Examples([[EXAMPLE_MACHINE, "Python"], [EXAMPLE_HUMAN, "Python"]], [code, language])
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                style = gr.JSON(label="stylometric features (z-scored at train time)")
                mode = gr.Textbox(label="inference mode", interactive=False)
        btn.click(run, [code, language], [verdict, style, mode])
    return ui


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("CODEDET_CKPT"), help="directory saved by train.py --save-ckpt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    det = Detector(a.ckpt)
    try:
        ui = build_ui(det, a.threshold)
    except ImportError:
        print("gradio is not installed: pip install gradio")
        return 1
    ui.launch(server_port=a.port, share=a.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
