"""
Export the trained CNN-BiLSTM-Attention model to ONNX for in-browser inference
with onnxruntime-web.

The architecture is re-declared inline (identical to web interface/app/model.py)
so this script has *no* mediapipe dependency and runs in any torch env.

Run:
    python static-site/tools/export_onnx.py

Outputs:
    static-site/assets/model/sign_model.onnx
    static-site/assets/model/class_labels.json   (copied)
    static-site/assets/model/model_meta.json
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # handle keypoints/
MODEL_DIR = ROOT / "models" / "alphabet"
OUT_DIR = ROOT / "static-site" / "assets" / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 45
NUM_FEATURES = 216


# --- architecture (mirror of CNNBiLSTMAttnModel) ---------------------------
class CNNBiLSTMAttnModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 30, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(192)
        self.dropout_conv = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=192, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.attn_proj = nn.Linear(256, 128)
        self.attn_score = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        attn_h = torch.tanh(self.attn_proj(x))
        attn_logits = self.attn_score(attn_h).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled = (x * attn_weights).sum(dim=1)
        pooled = self.dropout(pooled)
        # Emit probabilities so the JS side needs no softmax.
        return torch.softmax(self.fc(pooled), dim=1)


def main():
    labels_path = MODEL_DIR / "class_labels.json"
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    class_names = labels["classes"] if isinstance(labels, dict) else labels
    num_classes = len(class_names)

    model = CNNBiLSTMAttnModel(NUM_FEATURES, num_classes=num_classes, dropout=0.3)

    state = torch.load(MODEL_DIR / "best_model.pth", map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, SEQ_LEN, NUM_FEATURES, dtype=torch.float32)
    onnx_path = OUT_DIR / "sign_model.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["sequence"],
        output_names=["probs"],
        # Fixed batch=1: the browser always feeds exactly one (45,216) window,
        # and a fixed-shape LSTM exports/runs more reliably in onnxruntime-web.
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # use the stable TorchScript exporter (no onnxscript dep)
    )
    print(f"Wrote {onnx_path}")

    # Copy labels next to the model for the browser to fetch.
    with open(OUT_DIR / "class_labels.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names}, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"sequence_length": SEQ_LEN, "num_features": NUM_FEATURES,
             "num_classes": num_classes, "output": "probabilities"},
            f, ensure_ascii=False, indent=2,
        )

    # --- verify parity torch vs onnxruntime --------------------------------
    try:
        import onnxruntime as ort
        with torch.no_grad():
            torch_out = model(dummy).numpy()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"sequence": dummy.numpy()})[0]
        max_diff = float(np.max(np.abs(torch_out - ort_out)))
        print(f"max |torch - onnx| = {max_diff:.3e}")
        assert max_diff < 1e-4, "ONNX output diverges from torch!"
        print("ONNX parity OK")
    except ImportError:
        print("onnxruntime not installed — skipped parity check")


if __name__ == "__main__":
    main()
