import streamlit as st, pandas as pd, numpy as np, joblib, json
from pathlib import Path

# -------------------- Config --------------------
CONF_THRESHOLDS = dict(high=0.75, medium=0.55)
MODEL = Path("artifacts/model.joblib")
FEATS = ["orbital_period","transit_duration","transit_depth","planet_radius"]

SLIDER_LIMITS = {
    "orbital_period":   dict(min=0.0,   max=1000.0, step=0.1,  default=10.0,  unit="days"),
    "transit_duration": dict(min=0.0,   max=30.0,   step=0.05, default=3.0,   unit="hours"),
    "transit_depth":    dict(min=0.0,   max=20000.0,step=10.0, default=500.0, unit="ppm"),
    "planet_radius":    dict(min=0.0,   max=20.0,   step=0.1,  default=2.0,   unit="R_earth"),
}

# Rutas sugeridas para GIFs (déjalos vacíos o coloca tus archivos ahí)
GIF_MAP = {
    "CONFIRMED":      Path("artifacts/gifs/confirmed.gif"),
    "CANDIDATE":      Path("artifacts/gifs/candidate.gif"),
    "FALSE POSITIVE": Path("artifacts/gifs/false_positive.gif"),
}

# -------------------- Utils --------------------
def confidence_label(p: float) -> tuple[str, str]:
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

@st.cache_resource
def load_model():
    if not MODEL.exists():
        st.error("Not found: artifacts/model.joblib. Train first: python src/train_model.py")
        st.stop()
    return joblib.load(MODEL)

def prob_bars(classes, proba):
    order = np.argsort(proba)  # ascendente, la más alta queda abajo
    for i in order:
        st.write(f"{classes[i]} - {proba[i]:.0%}")
        st.progress(min(max(float(proba[i]), 0.0), 1.0))

def show_gif_for(pred_cls: str):
    path = GIF_MAP.get(pred_cls)
    if path and path.exists():
        st.image(str(path), caption=f"Visual: {pred_cls}", use_container_width=True)
    else:
        # Placeholder visual limpio
        st.markdown(
            f"""
            #### Visual (placeholder)
            _Drop a GIF at_: `{GIF_MAP.get(pred_cls)}`  
            - Expected for **{pred_cls}**.  
            - Recommended size: ~600-900px width, <5MB.
            """
        )
        st.write("")

# -------------------- UI --------------------
st.set_page_config(page_title="Exoplanet Classifier", page_icon="🪐", layout="wide")
st.title("🪐 Exoplanet Classifier (Kepler+TESS)")
st.caption("Clasifies as: CONFIRMED / CANDIDATE / FALSE POSITIVE")

model = load_model()

# -------- Input sliders --------
st.subheader("Manual entry")

with st.expander("What does each field mean?"):
    st.markdown("""
•  *Orbital period (days)*: how many days it takes the planet to orbit its star.  
•  *Transit duration (hours)*: duration of the transit in hours.  
•  *Transit depth (ppm)*: brightness drop during the transit (parts per million).  
•  *Planet radius (R_earth)*: planet radius in Earth radii (Earth = 1).
""")

c1, c2 = st.columns(2)
with c1:
    pconf = SLIDER_LIMITS["orbital_period"]
    period = st.slider("Orbital period (days)",
        min_value=float(pconf["min"]), max_value=float(pconf["max"]),
        value=float(pconf["default"]), step=float(pconf["step"]), help="Days"
    )
    dconf = SLIDER_LIMITS["transit_depth"]
    depth = st.slider("Transit depth (ppm)",
        min_value=float(dconf["min"]), max_value=float(dconf["max"]),
        value=float(dconf["default"]), step=float(dconf["step"]), help="ppm"
    )
with c2:
    tconf = SLIDER_LIMITS["transit_duration"]
    dur = st.slider("Transit duration (hours)",
        min_value=float(tconf["min"]), max_value=float(tconf["max"]),
        value=float(tconf["default"]), step=float(tconf["step"]), help="Hours"
    )
    rconf = SLIDER_LIMITS["planet_radius"]
    radius = st.slider("Planet radius (R_earth)",
        min_value=float(rconf["min"]), max_value=float(rconf["max"]),
        value=float(rconf["default"]), step=float(rconf["step"]), help="Earth radii"
    )

# -------- Predict --------
if st.button("Predict (manual)", use_container_width=True):
    X = pd.DataFrame([{
        "orbital_period": period,
        "transit_duration": dur,
        "transit_depth": depth,
        "planet_radius": radius
    }])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
    else:
        classes = list(model.classes_) if hasattr(model, "classes_") else ["CANDIDATE","CONFIRMED","FALSE POSITIVE"]
        proba = np.full(len(classes), 1/len(classes))

    pred_idx = int(np.argmax(proba))
    pred_cls = classes[pred_idx]
    pred_conf = float(proba[pred_idx])
    label, icon = confidence_label(pred_conf)

    st.success(f"*Prediction: {pred_cls}*  | Trust: *{pred_conf:.0%}* {icon}  ({label})")

    # --- 2 columnas: izquierda stats (50%), derecha GIF (50%) ---
    col_stats, col_gif = st.columns(2)

    with col_stats:
        st.markdown("### Probability distribution")
        prob_bars(classes, proba)

        st.markdown("### Input snapshot")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Period (d)", f"{period:.2f}")
        m2.metric("Dur (h)", f"{dur:.2f}")
        m3.metric("Depth (ppm)", f"{depth:.0f}")
        m4.metric("Radius (R⊕)", f"{radius:.2f}")

        with st.expander("See numerical details"):
            st.json({c: round(float(p), 6) for c, p in zip(classes, proba)})

        mets = read_metrics()
        if mets:
            macro_f1 = mets.get("report", {}).get("macro avg", {}).get("f1-score")
            if macro_f1 is not None:
                st.caption(f"Model performance (hold-out): Macro-F1 ≈ *{macro_f1:.2f}*")

    with col_gif:
        st.markdown("### Visual outcome")
        show_gif_for(pred_cls)

st.divider()

# -------- CSV --------
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
            # Asegura columnas ordenadas iguales a model.classes_
            for i, cls in enumerate(model.classes_ if hasattr(model, "classes_") else []):
                out[f"proba_{cls}"] = probas[:, i]
        st.dataframe(out.head(50), use_container_width=True)
        out.to_csv("artifacts/predictions.csv", index=False)
        st.success("Saved in artifacts/predictions.csv")

# -------- Footer hint --------
with st.expander("How to add your GIFs"):
    st.markdown(
        """
        Place your files here (create folders if needed):
        - `artifacts/gifs/confirmed.gif`
        - `artifacts/gifs/candidate.gif`
        - `artifacts/gifs/false_positive.gif`

        Filenames are customizable-solo actualiza `GIF_MAP` arriba.
        """
    )

