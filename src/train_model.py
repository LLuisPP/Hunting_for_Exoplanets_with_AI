import os, json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

DATA = "data/exoplanets.csv"
OUT_MODEL = "artifacts/model.joblib"
OUT_METRICS = "artifacts/metrics.json"
FEATS = ["orbital_period","transit_duration","transit_depth","planet_radius"]

df = pd.read_csv(DATA)
X, y = df[FEATS], df["label"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),  # opcional para RF
    ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1))
])

pipe.fit(Xtr, ytr)
yp = pipe.predict(Xte)

rep = classification_report(yte, yp, output_dict=True, zero_division=0)
cm = confusion_matrix(yte, yp).tolist()

os.makedirs("artifacts", exist_ok=True)
joblib.dump(pipe, OUT_MODEL)
json.dump({"report":rep, "confusion_matrix":cm, "features":FEATS, "labels":sorted(df['label'].unique())},
          open(OUT_METRICS,"w"), indent=2)

print(f"✅ Modelo: {OUT_MODEL}")
print(f"📊 Métricas: {OUT_METRICS}")
print(f"Macro F1: {rep['macro avg']['f1-score']:.3f}")
