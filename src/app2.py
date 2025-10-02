import streamlit as st, pandas as pd, numpy as np, joblib
from pathlib import Path
import json

CONF_THRESHOLDS = dict(high=0.75, medium=0.55)  # ajusta si quieres

def confidence_label(p: float) -> tuple[str, str]:
    """Devuelve (etiqueta, color) según el porcentaje p."""
    if p >= CONF_THRESHOLDS["high"]:
        return "Alta", "✅"
    if p >= CONF_THRESHOLDS["medium"]:
        return "Media", "🟡"
    return "Baja", "⚠️"

def read_metrics(path="artifacts/metrics.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

st.set_page_config(page_title="Exoplanet Classifier", page_icon="🪐", layout="centered")
st.title("🪐 Exoplanet Classifier (Kepler+TESS)")
st.caption("Clasifica: CONFIRMED / CANDIDATE / FALSE POSITIVE")

MODEL = Path("artifacts/model.joblib")
FEATS = ["orbital_period","transit_duration","transit_depth","planet_radius"]

@st.cache_resource
def load_model():
    if not MODEL.exists():
        st.error("No se encontró artifacts/model.joblib. Entrena primero: python src/train_model.py")
        st.stop()
    return joblib.load(MODEL)

model = load_model()

st.subheader("Entrada manual")
c1, c2 = st.columns(2)
with c1:
    period = st.number_input("Orbital period (days)", min_value=0.0, value=10.0, step=0.1)
    depth  = st.number_input("Transit depth (ppm)",   min_value=0.0, value=500.0, step=10.0)
with c2:
    dur    = st.number_input("Transit duration (hours)", min_value=0.0, value=3.0, step=0.1)
    radius = st.number_input("Planet radius (R_earth)",  min_value=0.0, value=2.0, step=0.1)

if st.button("Predecir (manual)"):
    X = pd.DataFrame([{
        "orbital_period": period,
        "transit_duration": dur,
        "transit_depth": depth,
        "planet_radius": radius
    }])
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    pred = classes[np.argmax(proba)]
    st.success(f"Predicción: **{pred}**")
    st.write({c: float(p) for c,p in zip(classes, proba)})

st.divider()
st.subheader("Predicción por CSV")
st.caption("Columnas: orbital_period, transit_duration, transit_depth, planet_radius")
uploaded = st.file_uploader("Sube un CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas: {missing}")
    else:
        preds = model.predict(df[FEATS])
        out = df.copy()
        out["prediction"] = preds
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(df[FEATS])
            for i, cls in enumerate(model.classes_):
                out[f"proba_{cls}"] = probas[:, i]
        st.dataframe(out.head(50))
        out.to_csv("artifacts/predictions.csv", index=False)
        st.success("Guardado en artifacts/predictions.csv")
