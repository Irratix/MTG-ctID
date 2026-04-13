from pathlib import Path
import torch
from torchvision.models import efficientnet_b0
import torch.nn as nn
import json

MODEL_INPUT_PATH = Path("models/checkpoint_epoch_1.pth")
MODEL_OUTPUT_PATH = Path("final_models/creature_classifier.onnx")

MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_INPUT_PATH, map_location=device, weights_only=False)
model = checkpoint['model']
model_types = checkpoint['all_types']
model = model.to(device)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    model,
    dummy_input,
    MODEL_OUTPUT_PATH,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=17
)

with open("final_models/all_types.json", "w") as f:
    json.dump(model_types, f)