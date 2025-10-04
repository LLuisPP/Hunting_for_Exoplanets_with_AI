import streamlit as st, pandas as pd, numpy as np, joblib, json
from pathlib import Path

# -------------------- Config --------------------
CONF_THRESHOLDS = dict(high=0.75, medium=0.55)
MODEL = Path("artifacts/model.joblib")
FEATS = ["orbital_period","transit_duration","transit_depth","planet_radius"]

SLIDER_LIMITS = {
    "orbital_period":   dict(min=0.0,   max=1000.0, step=0.1,  default=150.0,  unit="days"),
    "transit_duration": dict(min=0.0,   max=30.0,   step=0.05, default=5.0,   unit="hours"),
    "transit_depth":    dict(min=0.0,   max=20000.0,step=10.0, default=3000.0, unit="ppm"),
    "planet_radius":    dict(min=0.0,   max=40.0,   step=0.1,  default=7.0,   unit="R_earth"),
}

# (Solo para pequeños ajustes visuales que ayudan con la responsividad)

RESPONSIVE_CSS = """
<style>

/* === Tipografía fluida para st.metric en "Input snapshot" === */
/* Valores (números grandes) */
[data-testid="stMetricValue"] {
  /* De ~14px a ~22px según ancho de viewport */
  font-size: clamp(0.875rem, 2.2vw, 1.375rem) !important;
  line-height: 1.1 !important;
  white-space: nowrap;          /* evita saltos feos */
}

/* Etiquetas (Period, Dur, Depth, Radius) */
[data-testid="stMetricLabel"] {
  /* De ~11px a ~14px */
  font-size: clamp(0.70rem, 1.5vw, 0.875rem) !important;
  opacity: .9;
}

/* Delta (si lo usas en el futuro) */
[data-testid="stMetricDelta"] {
  font-size: clamp(0.70rem, 1.5vw, 0.875rem) !important;
}

/* Distribución flexible de las 4 métricas en pantallas pequeñas */
.metric-row { display: flex; gap: .6rem; flex-wrap: wrap; }
.metric-row > [data-testid="column"] { flex: 1 1 160px; }

/* En tablets: 2 por fila */
@media (max-width: 900px) {
  .metric-row > [data-testid="column"] { flex: 1 1 45%; }
}

/* En móviles estrechos: 1 por fila */
@media (max-width: 520px) {
  .metric-row > [data-testid="column"] { flex: 1 1 100%; }
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

RESPONSIVE_CSS += """
<style>
/* === COLOR AZUL GLOBAL (sliders y botones) === */
:root {
  --swai-blue: #007BFF;  /* puedes ajustar el tono de azul aquí */
}

/* Sliders (barra y círculo) */
div[data-testid="stSlider"] > div > div > div {
  background: var(--swai-blue) !important;    /* barra activa */
}
div[data-testid="stSlider"] [role="slider"] {
  background-color: var(--swai-blue) !important; /* botón circular */
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.3) !important;
}

/* Hover y foco para slider */
div[data-testid="stSlider"] [role="slider"]:hover {
  background-color: #339CFF !important;
}

/* Botones principales (Predict, etc.) */
button[kind="primary"],
.stButton > button {
  background-color: var(--swai-blue) !important;
  border: none !important;
  color: white !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  transition: background-color 0.2s ease-in-out;
}
button[kind="primary"]:hover,
.stButton > button:hover {
  background-color: #339CFF !important;
}

/* Bordes suaves en inputs */
input, textarea {
  border: 1px solid #aacdfd !important;
  border-radius: 6px !important;
}

</style>
"""


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
        help="Time for one full orbit around the star"
    )
    dconf = SLIDER_LIMITS["transit_depth"]
    depth = st.slider(
        "Transit depth (ppm)",
        min_value=float(dconf["min"]),
        max_value=float(dconf["max"]),
        value=float(dconf["default"]),
        step=float(dconf["step"]),
        help="Drop in starlight during a transit, in parts per million"
    )
with c2:
    tconf = SLIDER_LIMITS["transit_duration"]
    dur = st.slider(
        "Transit duration (hours)",
        min_value=float(tconf["min"]),
        max_value=float(tconf["max"]),
        value=float(tconf["default"]),
        step=float(tconf["step"]),
        help="Time the planet takes to cross the star"
    )
    rconf = SLIDER_LIMITS["planet_radius"]
    radius = st.slider(
        "Planet radius (R_earth)",
        min_value=float(rconf["min"]),
        max_value=float(rconf["max"]),
        value=float(rconf["default"]),
        step=float(rconf["step"]),
        help="Planet size in Earth radii proportion"
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
        st.markdown("#### Probability distribution")
        prob_bars(classes, proba)

        st.markdown("#### Input snapshot")
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
        st.markdown("#### Visual simulation")
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

