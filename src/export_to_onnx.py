import joblib, json
import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

MODEL_IN = "artifacts/model.joblib"
ONNX_OUT = "web/model.onnx"
META_OUT = "web/meta.json"

FEATURES = ["orbital_period","transit_duration","transit_depth","planet_radius"]

model = joblib.load(MODEL_IN)

# Tipo de entrada: [batch, num_features]
initial_types = [("input", FloatTensorType([None, len(FEATURES)]))]

onnx_model = to_onnx(model, initial_types=initial_types, target_opset=13)
# Guarda ONNX y metadatos
import os
os.makedirs("web", exist_ok=True)
with open(ONNX_OUT, "wb") as f:
    f.write(onnx_model.SerializeToString())

meta = {
    "features": FEATURES,
    "classes": list(model.classes_)  # orden de clases de predict_proba
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Exportado:", ONNX_OUT, META_OUT)

