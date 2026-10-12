import numpy as np
import pandas as pd
import pytest

import scanner as sc
import storage


def feats(grade, ore="Fresh", seed=1, exposure=1.0):
    img = sc.load_image(sc.synthetic_rock_image(grade, ore, np.random.default_rng(seed), exposure))
    return sc.extract_features(img)[0]


def test_segmentation_finds_rock_not_tray():
    f = feats(2.0)
    assert 0.07 < f["cover_frac"] < 0.35
    assert f["n_particles"] > 10


def test_dark_area_rises_with_grade():
    lo = np.mean([feats(0.5, seed=s)["dark_frac"] for s in range(4)])
    hi = np.mean([feats(5.0, seed=s)["dark_frac"] for s in range(4)])
    assert hi > lo + 0.1


def test_exposure_change_cancels_with_tray_reference():
    a, b = feats(3.0, seed=5, exposure=1.0), feats(3.0, seed=5, exposure=0.8)
    assert abs(a["dark_frac"] - b["dark_frac"]) < 0.04
    assert abs(a["mean_v"] - b["mean_v"]) < 0.04


def test_ore_colour_separates_types():
    assert feats(2, "Oxide")["rust_frac"] > feats(2, "Fresh")["rust_frac"] + 0.05


def test_quality_warnings():
    dark = np.full((200, 300, 3), 0.05)
    f = {"cover_frac": 0.5}
    assert any("dark" in w for w in sc.quality_warnings(dark, f))
    flat = np.full((200, 300, 3), 0.5)
    assert any("blurred" in w for w in sc.quality_warnings(flat, f))
    assert any("Very little rock" in w for w in sc.quality_warnings(flat, {"cover_frac": 0.01}))


def test_bad_file_raises():
    with pytest.raises(Exception):
        sc.analyse(b"not an image")


@pytest.fixture(scope="module")
def demo():
    return sc.synthetic_scans(45, seed=3)


def test_fit_refuses_without_enough_assays(demo):
    assert sc.fit(demo.head(10)) is None
    few = demo.copy()
    few.loc[few.index[5:], "assay_gpt"] = np.nan
    assert sc.fit(few) is None


def test_fit_predicts_held_out_scans(demo):
    m = sc.fit(demo)
    assert m["loo_r2"] > 0.6 and m["quality"] in ("Fair", "Good")
    train, test = demo.iloc[:35], demo.iloc[35:].dropna(subset=["assay_gpt"])
    mt = sc.fit(train)
    err = (sc.predict_all(mt, test) - test["assay_gpt"]).abs().mean()
    assert err < 0.8


def test_noise_labels_give_poor_quality(demo):
    rng = np.random.default_rng(0)
    bad = demo.copy()
    bad["assay_gpt"] = rng.permutation(bad["assay_gpt"].to_numpy())
    m = sc.fit(bad)
    assert m["quality"] == "Poor"


def test_predict_interval_and_cutoff(demo):
    m = sc.fit(demo)
    p = sc.predict(m, feats(5.0), cutoff_gpt=1.0)
    assert p["low"] <= p["grade_gpt"] <= p["high"] and p["call"] == "Above cut-off"
    p0 = sc.predict(m, feats(0.3), cutoff_gpt=6.0)
    assert p0["call"] == "Below cut-off"
    near = sc.predict(m, feats(2.5), cutoff_gpt=2.5)
    assert near["call"].startswith("Near")


def test_unfamiliar_photo_flagged(demo):
    m = sc.fit(demo)
    f = feats(2.0)
    f["dark_frac"] = 0.99
    assert sc.predict(m, f)["unfamiliar"]


def test_ore_type_guess(demo):
    guess = sc.classify_ore_type(demo, feats(2.0, "Oxide", seed=9))
    assert guess and guess[0] in ("Oxide", "Transition")
    assert sc.classify_ore_type(demo.head(2), feats(2.0)) is None


def test_scan_storage_roundtrip(tmp_path, demo):
    db = tmp_path / "t.db"
    storage.save_rows("scans", demo, path=str(db))
    back = storage.load_table("scans", path=str(db))
    assert len(back) == len(demo)
    rid = int(back.iloc[0]["id"])
    storage.update_cells("scans", rid, {"assay_gpt": 9.9, "bogus": 1}, path=str(db))
    assert storage.load_table("scans", path=str(db)).iloc[0]["assay_gpt"] == pytest.approx(9.9)
    storage.delete_ids("scans", [rid], path=str(db))
    assert len(storage.load_table("scans", path=str(db))) == len(demo) - 1
