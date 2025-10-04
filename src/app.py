import streamlit as st, pandas as pd, numpy as np, joblib
from pathlib import Path
import json

CONF_THRESHOLDS = dict(high=0.75, medium=0.55)  # ajusta si quieres

def confidence_label(p: float) -> tuple[str, str]:
    """Devuelve (etiqueta, color) según el porcentaje p."""
    if p >= CONF_THRESHOLDS["high"]:
        return "High", "✅"
    if p >= CONF_THRESHOLDS["medium"]:
        return "Medium", "🟡"
    return "Low", "⚠️"

def read_metrics(path="artifacts/metrics.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

st.set_page_config(page_title="Exoplanet Classifier", page_icon="🪐", layout="centered")
st.title("🪐 Exoplanet Classifier (Kepler+TESS)")
st.caption("Clasifies as: CONFIRMED / CANDIDATE / FALSE POSITIVE")

MODEL = Path("artifacts/model.joblib")
FEATS = ["orbital_period","transit_duration","transit_depth","planet_radius"]

@st.cache_resource
def load_model():
    if not MODEL.exists():
        st.error("Not found: artifacts/model.joblib. Train first: python src/train_model.py")
        st.stop()
    return joblib.load(MODEL)

model = load_model()

st.subheader("Manual entry")

# Glosario breve para hacerlo más intuitivo
with st.expander("What does each field mean?"):
    st.markdown("""
•⁠  ⁠*Orbital period (days)*: how many days it takes the planet to orbit its star.  
•⁠  ⁠*Transit duration (hours)*: duration of transit (power outage) in hours.  
•⁠  ⁠*Transit depth (ppm)*: gloss decay during transit (parts per million)).  
•⁠  ⁠*Planet radius (R_earth)*: radius of the planet in terrestrial radios (Earth = 1).
""")

c1, c2 = st.columns(2)
with c1:
    period = st.number_input("Orbital period (days)", min_value=0.0, value=10.0, step=0.1, help="Days")
    depth  = st.number_input("Transit depth (ppm)",   min_value=0.0, value=500.0, step=10.0, help="ppm")
with c2:
    dur    = st.number_input("Transit duration (hours)", min_value=0.0, value=3.0, step=0.1, help="Hours")
    radius = st.number_input("Planet radius (R_earth)",  min_value=0.0, value=2.0, step=0.1, help="Radius terrestres")

if st.button("Predict (manual)"):
    X = pd.DataFrame([{
        "orbital_period": period,
        "transit_duration": dur,
        "transit_depth": depth,
        "planet_radius": radius
    }])

    # Probabilidades
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
    else:
        # Fallback (muy raro con RandomForest): probas uniformes
        classes = list(model.classes_) if hasattr(model, "classes_") else ["CANDIDATE","CONFIRMED","FALSE POSITIVE"]
        proba = np.full(len(classes), 1/len(classes))

    pred_idx = int(np.argmax(proba))
    pred_cls = classes[pred_idx]
    pred_conf = float(proba[pred_idx])

    # Semáforo de confianza
    label, icon = confidence_label(pred_conf)

    # Cabecera resultado
    st.success(f"*Prediction: {pred_cls}*  | Trust: *{pred_conf:.0%}* {icon}  ({label})")

    # Visualización: barras horizontales de probabilidad
    st.write("*Probability distribution*")
    order = np.argsort(proba)  # ascendente para que la barra grande quede abajo
    for i in order:
        st.write(f"{classes[i]} - {proba[i]:.0%}")
        st.progress(min(max(float(proba[i]), 0.0), 1.0))

    # Detalles numéricos (transparencia)
    with st.expander("See numerical details"):
        st.json({c: round(float(p), 6) for c, p in zip(classes, proba)})

    # Métricas del modelo (si existen)
    mets = read_metrics()
    if mets:
        macro_f1 = mets.get("report", {}).get("macro avg", {}).get("f1-score")
        if macro_f1 is not None:
            st.caption(f"Model performance (hold-out): Macro-F1 ≈ *{macro_f1:.2f}*")


st.divider()
st.subheader("Prediction by CSV")
st.caption("Columns: orbital_period, transit_duration, transit_depth, planet_radius")
uploaded = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        st.error(f"Columns are missing: {missing}")
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
        st.success("Saved in artifacts/predictions.csv")
