import pandas as pd

def fetch_koi():
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        "?table=koi&format=csv"
        "&select=koi_disposition,koi_pdisposition,koi_period,koi_duration,koi_depth,koi_prad"
    )
    koi = pd.read_csv(url)
    label = koi["koi_disposition"].fillna(koi["koi_pdisposition"])
    koi = koi.assign(label=label).rename(columns={
        "koi_period":"orbital_period",
        "koi_duration":"transit_duration",
        "koi_depth":"transit_depth",
        "koi_prad":"planet_radius",
    })[["orbital_period","transit_duration","transit_depth","planet_radius","label"]]
    koi = koi[koi["label"].isin(["CONFIRMED","CANDIDATE","FALSE POSITIVE"])]
    return koi

def fetch_toi():
    query = "select+tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep,pl_rade+from+toi"
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=csv"
    toi = pd.read_csv(url)
    map_disp = {"CP":"CONFIRMED","PC":"CANDIDATE","FP":"FALSE POSITIVE","FA":"FALSE POSITIVE"}
    toi["label"] = toi["tfopwg_disp"].map(map_disp)
    toi = toi.rename(columns={
        "pl_orbper":"orbital_period",
        "pl_trandurh":"transit_duration",
        "pl_trandep":"transit_depth",
        "pl_rade":"planet_radius",
    })[["orbital_period","transit_duration","transit_depth","planet_radius","label"]]
    toi = toi.dropna(subset=["label"])
    return toi

if __name__ == "__main__":
    df = pd.concat([fetch_koi(), fetch_toi()], ignore_index=True)
    df = df.dropna(subset=["orbital_period","transit_duration","transit_depth","planet_radius","label"])
    df.to_csv("data/exoplanets.csv", index=False)
    print(f"✅ Dataset guardado en data/exoplanets.csv | filas: {len(df)}")
