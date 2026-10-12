"""Ore grade scanner: measure a photo of crushed rock and estimate grade from the plant's OWN calibration.

Gold is far too fine to see in a photo. What a camera CAN see is what travels with gold on a given
deposit: sulphide darkness, iron-oxide staining, quartz veining, colour, texture. The relationship
between those and the assay is different on every mine, so the app never guesses it. It measures the
photo, then learns the link from scans that have a lab assay (ridge regression, checked leave-one-out).
Until enough assayed scans exist, it shows the measurements only and refuses to give a grade.

Photo method (must stay the same every time): rock spread in a single layer on a plain, uniform tray,
the same light (a box or fixed lamp), the same camera height, a dry sample. The tray is used as the
colour/exposure reference, so changes in light level cancel out.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage as ndi

MAX_SIDE = 640
MIN_CAL_SCANS = 15
MODEL_FEATURES = ["mean_v", "std_v", "mean_s", "r_frac", "b_frac", "dark_frac", "light_frac",
                  "rust_frac", "texture"]
FEATURE_LABELS = {
    "mean_v": "Brightness", "std_v": "Brightness spread", "mean_s": "Colour strength",
    "r_frac": "Redness", "b_frac": "Blueness", "dark_frac": "Dark mineral area (sulphides)",
    "light_frac": "Light mineral area (quartz)", "rust_frac": "Rust / iron staining",
    "texture": "Surface texture", "cover_frac": "Rock cover of tray", "n_particles": "Fragments counted",
    "d50_rel": "Typical fragment size (% of photo width)",
}
FEATURE_COLUMNS = list(FEATURE_LABELS)


# ───────────────────────── reading the photo ─────────────────────────
def load_image(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    im.thumbnail((max_side, max_side))
    return np.asarray(im, dtype=float) / 255.0


def _otsu(x: np.ndarray, bins: int = 64) -> float:
    h, edges = np.histogram(x, bins=bins)
    p = h / max(h.sum(), 1)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * mids)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0) ** 2 / (w0 * (1 - w0))
    between = np.nan_to_num(between)
    return float(mids[int(np.argmax(between))])


def segment(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rock mask (True = rock) and the tray colour. The tray colour is taken from the photo's border."""
    h, w, _ = img.shape
    b = max(2, int(0.05 * min(h, w)))
    border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                             img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((img - bg) ** 2).sum(axis=2))
    dist = ndi.gaussian_filter(dist, 1.0)
    thr = max(0.08, _otsu(dist.ravel()) * 0.8)
    mask = dist > thr
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    return mask, bg


def quality_warnings(img: np.ndarray, feats: dict) -> list[str]:
    out = []
    grey = img.mean(axis=2)
    lap = ndi.laplace(ndi.gaussian_filter(grey, 0.7))
    if lap.var() < 2e-5:
        out.append("The photo looks blurred. Hold the camera steady and focus on the rock.")
    if grey.mean() < 0.15:
        out.append("The photo is very dark. Add light or use the light box.")
    if (grey > 0.97).mean() > 0.10:
        out.append("Parts of the photo are over-exposed (glare). Move the light or tilt it away.")
    if feats["cover_frac"] < 0.05:
        out.append("Very little rock was found. Fill more of the tray, or use a tray that contrasts with the rock.")
    if feats["cover_frac"] > 0.92:
        out.append("The rock covers the whole photo. Leave a visible strip of empty tray around the edge.")
    return out


def extract_features(img: np.ndarray, tray_reference: bool = True) -> tuple[dict, np.ndarray]:
    """Return (features, rock mask). Colours are measured relative to the empty tray when tray_reference."""
    mask, bg = segment(img)
    cover = float(mask.mean())
    if cover < 0.02:
        mask = np.ones(mask.shape, dtype=bool)            # nothing found: measure everything, warn later
    norm = np.clip(img / np.maximum(bg, 1e-3) * 0.6, 0, 1) if tray_reference else img
    px = norm[mask]
    hsv = rgb_to_hsv(px)
    v, s, hue = hsv[:, 2], hsv[:, 1], hsv[:, 0]
    tot = px.sum(axis=1) + 1e-9
    grey = norm.mean(axis=2)
    lap = np.abs(ndi.laplace(ndi.gaussian_filter(grey, 0.7)))[mask]
    lab, n = ndi.label(mask)
    if n:
        areas = np.bincount(lab.ravel())[1:]
        areas = areas[areas >= 40]
    else:
        areas = np.array([])
    if len(areas):
        diam = 2 * np.sqrt(areas / np.pi)
        order = np.argsort(diam)
        cum = np.cumsum(areas[order]) / areas.sum()
        d50 = float(diam[order][np.searchsorted(cum, 0.5)] / img.shape[1] * 100)
    else:
        d50 = float("nan")
    f = {
        "mean_v": float(v.mean()), "std_v": float(v.std()), "mean_s": float(s.mean()),
        "r_frac": float((px[:, 0] / tot).mean()), "b_frac": float((px[:, 2] / tot).mean()),
        "dark_frac": float((v < 0.25).mean()), "light_frac": float((v > 0.75).mean()),
        "rust_frac": float(((hue > 0.02) & (hue < 0.12) & (s > 0.35) & (v > 0.2)).mean()),
        "texture": float(lap.mean() * 10),
        "cover_frac": cover, "n_particles": float(len(areas)), "d50_rel": d50,
    }
    return f, mask


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Photo with the detected rock outlined in orange, so the user can see what was measured."""
    edge = mask ^ ndi.binary_erosion(mask, iterations=2)
    out = img.copy()
    out[edge] = [0.92, 0.41, 0.20]
    return out


def analyse(data: bytes, tray_reference: bool = True) -> dict:
    img = load_image(data)
    feats, mask = extract_features(img, tray_reference)
    return {"image": img, "mask": mask, "features": feats, "warnings": quality_warnings(img, feats)}


# ───────────────────────── learning the grade ─────────────────────────
def labelled(scans: pd.DataFrame) -> pd.DataFrame:
    need = MODEL_FEATURES + ["assay_gpt"]
    if scans.empty or any(c not in scans.columns for c in need):
        return pd.DataFrame(columns=need)
    x = scans.copy()
    for c in need:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=need)


def fit(scans: pd.DataFrame, min_scans: int = MIN_CAL_SCANS) -> dict | None:
    """Ridge regression of assay on photo features. Penalty chosen by leave-one-out error."""
    d = labelled(scans)
    if len(d) < min_scans:
        return None
    X = d[MODEL_FEATURES].to_numpy(float)
    y = d["assay_gpt"].to_numpy(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    best = None
    for lam in (0.01, 0.1, 0.3, 1, 3, 10, 30, 100):
        A = np.linalg.inv(Z.T @ Z + lam * np.eye(Z.shape[1]))
        H = Z @ A @ Z.T
        res = (y - ym) - H @ (y - ym)
        loo = res / (1 - np.diag(H) - 1.0 / len(y))          # leave-one-out residuals (incl. centring)
        rmse = float(np.sqrt(np.mean(loo ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, lam, A @ Z.T @ (y - ym), loo)
    rmse, lam, w, loo = best
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1 - float((loo ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"mu": mu, "sd": sd, "w": w, "ym": float(ym), "lam": lam, "n": len(d), "loo_rmse": rmse,
            "loo_r2": r2, "loo_pred": y - loo, "loo_actual": y, "features": MODEL_FEATURES,
            "quality": quality_label(r2)}


def quality_label(r2: float) -> str:
    if not np.isfinite(r2) or r2 < 0.4:
        return "Poor"
    return "Fair" if r2 < 0.7 else "Good"


def predict(model: dict, feats: dict, cutoff_gpt: float | None = None) -> dict:
    z = (np.array([feats[c] for c in model["features"]], float) - model["mu"]) / model["sd"]
    est = max(0.0, float(model["ym"] + z @ model["w"]))
    err = 1.96 * model["loo_rmse"]
    out = {"grade_gpt": est, "low": max(0.0, est - err), "high": est + err,
           "unfamiliar": bool(np.abs(z).max() > 3.5)}
    if cutoff_gpt is not None:
        if out["low"] >= cutoff_gpt:
            out["call"] = "Above cut-off"
        elif out["high"] <= cutoff_gpt:
            out["call"] = "Below cut-off"
        else:
            out["call"] = "Near cut-off: send to the lab"
    return out


def predict_all(model: dict | None, scans: pd.DataFrame) -> pd.Series:
    if model is None or scans.empty:
        return pd.Series(np.nan, index=scans.index)
    ok = scans[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    Z = (ok.to_numpy(float) - model["mu"]) / model["sd"]
    return pd.Series(np.clip(model["ym"] + Z @ model["w"], 0, None), index=scans.index)


def classify_ore_type(scans: pd.DataFrame, feats: dict, min_each: int = 3) -> tuple[str, float] | None:
    """Nearest-centroid guess of the ore type from colour/texture, using the plant's own labelled scans."""
    d = scans.dropna(subset=["ore_type"]) if "ore_type" in scans.columns else scans.iloc[0:0]
    d = d[d["ore_type"].astype(str).str.len() > 0]
    d = d.dropna(subset=MODEL_FEATURES)
    counts = d["ore_type"].value_counts()
    keep = counts[counts >= min_each].index
    if len(keep) < 2:
        return None
    d = d[d["ore_type"].isin(keep)]
    X = d[MODEL_FEATURES].to_numpy(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    cent = {t: ((g[MODEL_FEATURES].to_numpy(float) - mu) / sd).mean(axis=0) for t, g in d.groupby("ore_type")}
    z = (np.array([feats[c] for c in MODEL_FEATURES], float) - mu) / sd
    dist = {t: float(np.linalg.norm(z - c)) for t, c in cent.items()}
    best = min(dist, key=dist.get)
    return best, dist[best]


# ───────────────────────── synthetic photos (demo and tests only) ─────────────────────────
_BASE = {"Oxide": (150, 105, 65), "Transition": (128, 112, 95), "Fresh": (108, 108, 114)}


def synthetic_rock_image(grade_gpt: float, ore_type: str = "Fresh", rng=None, exposure: float = 1.0,
                         size=(480, 360)) -> bytes:
    """A made-up photo: grey tray, rock fragments, dark 'sulphide' speckle that grows with grade.
    The grade-darkness link here is invented so the demo can calibrate. It is NOT a real ore."""
    rng = rng or np.random.default_rng()
    w, h = size
    tray = (172, 172, 168)
    im = Image.new("RGB", size, tray)
    dr = ImageDraw.Draw(im)
    base = np.array(_BASE.get(ore_type, _BASE["Fresh"]), float)
    speckle = min(0.6, 0.02 + 0.07 * grade_gpt)
    n_frag = int(rng.integers(35, 55))
    for _ in range(n_frag):
        cx, cy = rng.uniform(25, w - 25), rng.uniform(25, h - 25)
        r = rng.uniform(10, 24)
        ang = np.sort(rng.uniform(0, 2 * np.pi, 7))
        rad = r * rng.uniform(0.6, 1.1, 7)
        pts = [(cx + rr * np.cos(a), cy + rr * np.sin(a)) for rr, a in zip(rad, ang)]
        col = tuple(int(np.clip(c * rng.uniform(0.85, 1.12), 0, 255)) for c in base)
        dr.polygon(pts, fill=col)
        for _ in range(int(speckle * 40)):                 # dark mineral grains inside the fragment
            px, py = cx + rng.uniform(-r, r) * 0.7, cy + rng.uniform(-r, r) * 0.7
            g = rng.uniform(1.5, 3.5)
            dr.ellipse([px - g, py - g, px + g, py + g], fill=(28, 28, 32))
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(im, float)
    yy, xx = np.mgrid[0:h, 0:w]
    a *= (1 + 0.06 * (xx / w - 0.5))[..., None]            # uneven lighting
    a += rng.normal(0, 3, a.shape)
    a = np.clip(a * exposure, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def synthetic_scans(n: int = 60, seed: int = 11, start="2026-06-02") -> pd.DataFrame:
    """Demo scan history built by running the real analysis on synthetic photos. Grades are fictitious."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range(start, periods=n, freq="2D")
    for i in range(n):
        ore = str(rng.choice(["Oxide", "Transition", "Fresh"]))
        mean = {"Oxide": 1.8, "Transition": 2.6, "Fresh": 3.4}[ore]
        grade = float(np.clip(rng.gamma(4.0, mean / 4.0), 0.2, 8.0))
        f, _ = extract_features(load_image(synthetic_rock_image(grade, ore, rng, rng.uniform(0.8, 1.15))))
        lab = round(grade * float(rng.normal(1, 0.06)), 3) if i % 4 != 3 else np.nan   # some not yet assayed
        rows.append({"date": dates[i], "sample_id": f"DEMO-{i + 1:03d}", "ore_type": ore,
                     "location": "Stockpile " + str(rng.choice(["A", "B", "C"])), "assay_gpt": lab,
                     "note": "synthetic demo scan", **f})
    return pd.DataFrame(rows)
