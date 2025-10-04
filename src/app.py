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

# (Solo para pequeños ajustes visuales que ayudan con la responsividad)
RESPONSIVE_CSS = """
<style>
/* Reduce padding lateral en pantallas estrechas para aprovechar ancho */
@media (max-width: 640px) {
  .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
}

/* Evita que los metric queden aplastados en móviles */
.metric-row { display: flex; gap: .5rem; flex-wrap: wrap; }
.metric-row > div { flex: 1 1 110px; }

/* Títulos ligeramente más compactos en móvil */
@media (max-width: 480px) {
  h1, h2, h3 { line-height: 1.15; }
}
</style>
"""

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

def _normalize_option_from_class(pred_cls: str) -> str:
    """
    Map model class label -> option folder name:
    'CONFIRMED' -> 'confirmed'
    'CANDIDATE' -> 'candidate'
    'FALSE POSITIVE' -> 'false_positive'
    """
    return pred_cls.strip().lower().replace(" ", "_")

def find_gif_path_for(pred_cls: str) -> Path | None:
    """
    Busca el primer archivo animado disponible dentro de artifacts/gifs/<option>/
    Se aceptan .gif, .webp y .mp4.
    """
    option = _normalize_option_from_class(pred_cls)  # confirmed | candidate | false_positive
    base = Path("artifacts/gifs") / option
    if not base.exists():
        return None
    exts = (".gif", ".webp", ".mp4")
    for ext in exts:
        files = sorted(base.glob(f"*{ext}"))
        if files:
            return files[0]
    return None

def show_media_for(pred_cls: str):
    """
    Carga el media de forma robusta y responsive:
    - GIF/WEBP -> se leen como bytes y se envían a st.image(use_container_width=True)
    - MP4 -> se pasa ruta absoluta a st.video
    """
    media = find_gif_path_for(pred_cls)
    if not media:
        st.info("No media found for this class.")
        return

    suffix = media.suffix.lower()
    if suffix == ".mp4":
        st.video(str(media.resolve()))
        return

    # GIF/WEBP como bytes para evitar problemas de rutas/servido estático
    try:
        data = media.read_bytes()
        st.image(data, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load media: {media.name}")
        st.exception(e)

# -------------------- UI --------------------
st.set_page_config(page_title="SWAI - Exoplanet Classifier", page_icon="🪐", layout="wide")
st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)

st.title("Silent Watcher AI 🪐 Exoplanet Classifier")
st.caption("Categories: CONFIRMED / CANDIDATE / FALSE POSITIVE based on NASA's Kepler/TESS datasets")

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
    period = st.slider(
        "Orbital period (days)",
        min_value=float(pconf["min"]),
        max_value=float(pconf["max"]),
        value=float(pconf["default"]),
        step=float(pconf["step"]),
        help="Days"
    )
    dconf = SLIDER_LIMITS["transit_depth"]
    depth = st.slider(
        "Transit depth (ppm)",
        min_value=float(dconf["min"]),
        max_value=float(dconf["max"]),
        value=float(dconf["default"]),
        step=float(dconf["step"]),
        help="ppm"
    )
with c2:
    tconf = SLIDER_LIMITS["transit_duration"]
    dur = st.slider(
        "Transit duration (hours)",
        min_value=float(tconf["min"]),
        max_value=float(tconf["max"]),
        value=float(tconf["default"]),
        step=float(tconf["step"]),
        help="Hours"
    )
    rconf = SLIDER_LIMITS["planet_radius"]
    radius = st.slider(
        "Planet radius (R_earth)",
        min_value=float(rconf["min"]),
        max_value=float(rconf["max"]),
        value=float(rconf["default"]),
        step=float(rconf["step"]),
        help="Earth radii"
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

    # --- 2 columnas: izquierda stats (50%), derecha media (50%) ---
    col_stats, col_media = st.columns(2, gap="large")

    with col_stats:
        st.markdown("### Probability distribution")
        prob_bars(classes, proba)

        st.markdown("### Input snapshot")
        # Fila responsive para metrics (CSS arriba)
        with st.container():
            st.markdown('<div class="metric-row">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Period (d)", f"{period:.2f}")
            m2.metric("Dur (h)", f"{dur:.2f}")
            m3.metric("Depth (ppm)", f"{depth:.0f}")
            m4.metric("Radius (R⊕)", f"{radius:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("See numerical details"):
            st.json({c: round(float(p), 6) for c, p in zip(classes, proba)})

        mets = read_metrics()
        if mets:
            macro_f1 = mets.get("report", {}).get("macro avg", {}).get("f1-score")
            if macro_f1 is not None:
                st.caption(f"Model performance (hold-out): Macro-F1 ≈ *{macro_f1:.2f}*")

    with col_media:
        st.markdown("### Visual cue")
        # Carga robusta del media, ocupa el ancho disponible (responsive)
        show_media_for(pred_cls)

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
            if hasattr(model, "classes_"):
                for i, cls in enumerate(model.classes_):
                    out[f"proba_{cls}"] = probas[:, i]
        st.dataframe(out.head(50), use_container_width=True)
        out.to_csv("artifacts/predictions.csv", index=False)
        st.success("Saved in artifacts/predictions.csv")

