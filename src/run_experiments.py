"""Runner eksperimen (P4): menjalankan recognition untuk setiap skenario uji.

Alur run_all():
    scan_dataset -> check_dataset -> extract_all -> build_systems (sekali, hanya
    dari data/template) -> run_scenario untuk tiap skenario -> results/predictions.csv

Pemakaian:
    python -m src.run_experiments --use-stubs          # sebelum kode P1-P3 siap
    python -m src.run_experiments                      # pakai kode asli P1-P3
    python -m src.run_experiments --scenarios closed_speaker --systems gaussian

Kontrak fungsi yang dipanggil dari P1-P3 (lihat juga src/_stubs.py):
    src.features.extract_mfcc39(path) -> (T, 39)
    src.template.build_averaged_template(list_of_(T_i, 39)) -> (T, 39)
    src.dtw.dtw_distance(X, Y) -> float
    src.gaussian.fit_gaussian(list_of_(T_i, 39)) -> model
    src.gaussian.nll(X, model) -> float
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import soundfile as sf

from src.recognize import recognize_dtw, recognize_dtw_nn, recognize_gaussian

VOWELS = ["A", "I", "U", "E", "O"]
SCENARIO_DIRS = {
    "template": "template",
    "closed_recording": "test/closed/recording",
    "closed_speaker": "test/closed/speaker",
    "open": "test/open",
}
TEST_SCENARIOS = ["closed_recording", "closed_speaker", "open", "loso"]
DEFAULT_SCENARIOS = ["closed_recording", "closed_speaker", "open"]
SYSTEMS = ["dtw", "gaussian", "dtw_nn"]
DEFAULT_SYSTEMS = ["dtw", "gaussian"]

# Konvensi nama file: S{speaker}_{vowel}_{usage}_{condition}.wav, contoh S01_A_CS_HP.wav
FILENAME_RE = re.compile(r"^(?P<speaker>[A-Za-z0-9]+)_(?P<vowel>[AIUEO])_(?P<usage>CR|CS|OS)_(?P<condition>[A-Za-z]+)\.wav$", re.IGNORECASE)
EXPECTED_SAMPLE_RATE = 16000
MAX_DURATION = 10.0  # detik; rekaman lebih panjang hampir pasti belum dipotong per vowel


def load_backend(use_stubs=False):
    """Ambil fungsi MFCC, template, DTW, dan Gaussian dari P1-P3 (atau stub sementara)."""
    if use_stubs:
        from src import _stubs as s

        return SimpleNamespace(name="stubs", extract=s.extract_mfcc39, build_template=s.build_averaged_template,
                               dtw=s.dtw_distance, fit_gaussian=s.fit_gaussian, nll=s.nll)
    try:
        from src.dtw import dtw_distance
        from src.features import extract_mfcc_39
        from src.gaussian import fit_gaussian, nll
        from src.preprocessing import load_audio
        from src.template import build_averaged_template
    except ImportError as e:
        raise ImportError(f"Kode P1-P3 belum lengkap ({e}). Untuk sementara jalankan dengan --use-stubs.") from e
    def extract_mfcc39(path):
        # konfigurasi sama dengan main.ipynb: WINLEN=0.025, WINSTEP=0.010, NUM_CEP=13, N_FFT=512
        signal, sample_rate = load_audio(path, EXPECTED_SAMPLE_RATE)
        return extract_mfcc_39(signal, sample_rate, 0.025, 0.010, 13, 512)

    return SimpleNamespace(name="p1-p3", extract=extract_mfcc39, build_template=build_averaged_template,
                           dtw=dtw_distance, fit_gaussian=fit_gaussian, nll=nll)


def scan_dataset(data_dir="data"):
    """Daftar semua file .wav di data/ beserta label, skenario, hash, dan info audio."""
    rows = []
    for scenario, sub in SCENARIO_DIRS.items():
        for path in sorted((Path(data_dir) / sub).rglob("*.wav")):
            m = FILENAME_RE.match(path.name)
            info = sf.info(str(path))
            rows.append({
                "path": str(path),
                "file": path.name,
                "scenario": scenario,
                "folder_vowel": path.parent.name.upper() if path.parent.name.upper() in VOWELS else None,
                "speaker": m["speaker"].upper() if m else None,
                "vowel": m["vowel"].upper() if m else None,
                "usage": m["usage"].upper() if m else None,
                "condition": m["condition"].upper() if m else None,
                "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration": info.duration,
            })
    return pd.DataFrame(rows, columns=["path", "file", "scenario", "folder_vowel", "speaker", "vowel", "usage",
                                       "condition", "md5", "sample_rate", "channels", "duration"])


def check_dataset(df, strict=True):
    """Pastikan pengujian adil dan label benar. Error -> ValueError (kecuali strict=False).

    Returns:
        Daftar peringatan yang tidak menghentikan eksperimen.
    """
    errors, warnings = [], []
    by = {s: df[df.scenario == s] for s in SCENARIO_DIRS}
    template = by["template"]

    for f in df[df.vowel.isna()].path:
        errors.append(f"Nama file tidak sesuai konvensi S??_V_USAGE_KONDISI.wav: {f}")
    named = df.dropna(subset=["vowel"])
    for f in named[named.folder_vowel.notna() & (named.folder_vowel != named.vowel)].path:
        errors.append(f"Label folder berbeda dengan label nama file: {f}")
    for f in df[df.duration > MAX_DURATION].path:
        errors.append(f"Durasi > {MAX_DURATION:.0f} detik, kemungkinan belum dipotong per vowel: {f}")
    for f in df[(df.sample_rate != EXPECTED_SAMPLE_RATE) | (df.channels != 1)].path:
        warnings.append(f"Bukan {EXPECTED_SAMPLE_RATE} Hz mono (pastikan ekstraksi fitur menanganinya): {f}")

    # Template: kelima vowel ada, dan speakernya sama untuk setiap vowel
    speakers_per_vowel = {v: set(template[template.vowel == v].speaker) for v in VOWELS}
    for v, spk in speakers_per_vowel.items():
        if not spk:
            errors.append(f"Tidak ada data template untuk vowel {v}")
        elif len(spk) != 5:
            warnings.append(f"Template vowel {v} berisi {len(spk)} speaker (spek: 5)")
    if len({frozenset(s) for s in speakers_per_vowel.values() if s}) > 1:
        errors.append(f"Speaker template tidak sama di setiap vowel: { {v: sorted(s) for v, s in speakers_per_vowel.items()} }")
    template_speakers = set(template.speaker.dropna())

    # File identik tidak boleh punya label berbeda atau muncul dua kali dalam satu skenario
    for md5, group in named.groupby("md5"):
        if group.vowel.nunique() > 1:
            errors.append(f"File identik dengan label vowel berbeda: {sorted(group.path)}")
        for scenario, g in group.groupby("scenario"):
            if len(g) > 1:
                errors.append(f"File identik muncul {len(g)}x di {scenario}: {sorted(g.path)}")

    # Closed recording = rekaman template itu sendiri
    key = ["speaker", "vowel", "md5"]
    if set(map(tuple, by["closed_recording"][key].values)) != set(map(tuple, template[key].values)):
        errors.append("Isi test/closed/recording tidak identik dengan data/template")

    # Closed speaker = speaker template, rekaman berbeda
    leaked = by["closed_speaker"][by["closed_speaker"].md5.isin(set(template.md5))]
    for f in leaked.path:
        errors.append(f"Data closed speaker identik dengan rekaman template: {f}")
    for spk in sorted(set(by["closed_speaker"].speaker.dropna()) - template_speakers):
        errors.append(f"Speaker {spk} di closed speaker tidak ada di template")

    # Open = speaker di luar template
    for spk in sorted(set(by["open"].speaker.dropna()) & template_speakers):
        errors.append(f"Speaker {spk} di open juga ada di template (seharusnya speaker lain)")

    for scenario in ["template", "closed_recording", "closed_speaker"]:
        if by[scenario].empty:
            errors.append(f"Tidak ada file untuk {scenario}")
    if by["open"].empty:
        warnings.append("Folder test/open kosong: skenario open dilewati")

    if errors and strict:
        raise ValueError("Dataset tidak lolos pengecekan:\n- " + "\n- ".join(errors))
    return warnings + errors


def extract_all(df, extract_fn):
    """Ekstraksi MFCC sekali per file; file identik (hash sama) hanya diekstrak sekali."""
    by_md5 = {}
    for path, md5 in zip(df.path, df.md5):
        if md5 not in by_md5:
            by_md5[md5] = extract_fn(path)
    return {path: by_md5[md5] for path, md5 in zip(df.path, df.md5)}


def build_systems(train_df, feats, backend, systems=DEFAULT_SYSTEMS):
    """Bangun model tiap sistem dari data latih (normalnya: hanya data/template)."""
    by_vowel = {v: [feats[p] for p in train_df[train_df.vowel == v].path] for v in VOWELS}
    builders = {
        "dtw": lambda: {v: backend.build_template(fs) for v, fs in by_vowel.items()},
        "gaussian": lambda: {v: backend.fit_gaussian(fs) for v, fs in by_vowel.items()},
        "dtw_nn": lambda: by_vowel,
    }
    return {name: builders[name]() for name in systems}


RECOGNIZERS = {
    "dtw": lambda X, models, backend: recognize_dtw(X, models, backend.dtw),
    "gaussian": lambda X, models, backend: recognize_gaussian(X, models, backend.nll),
    "dtw_nn": lambda X, models, backend: recognize_dtw_nn(X, models, backend.dtw),
}


def _evaluate(test_df, feats, trained, backend, scenario):
    rows = []
    for item in test_df.itertuples():
        X = feats[item.path]
        for system, models in trained.items():
            start = time.perf_counter()
            pred, scores, margin = RECOGNIZERS[system](X, models, backend)
            rows.append({
                "file": item.file,
                "speaker": item.speaker,
                "vowel_true": item.vowel,
                "condition": item.condition,
                "scenario": scenario,
                "system": system,
                "pred": pred,
                "correct": int(pred == item.vowel),
                **{f"score_{v}": scores[v] for v in VOWELS},
                "margin": margin,
                "n_frames": len(X),
                "time_ms": (time.perf_counter() - start) * 1000,
            })
    return rows


def run_scenario(scenario, df, feats, trained, backend):
    """Jalankan satu skenario uji dan kembalikan DataFrame prediksi.

    "loso" (leave-one-speaker-out) melatih ulang tanpa satu speaker template,
    lalu menguji data closed speaker milik speaker tersebut, bergiliran.
    """
    if scenario == "loso":
        template, closed = df[df.scenario == "template"], df[df.scenario == "closed_speaker"]
        rows = []
        for spk in sorted(template.speaker.unique()):
            fold = build_systems(template[template.speaker != spk], feats, backend, list(trained))
            rows += _evaluate(closed[closed.speaker == spk], feats, fold, backend, scenario)
    else:
        rows = _evaluate(df[df.scenario == scenario], feats, trained, backend, scenario)
    return pd.DataFrame(rows)


def run_all(data_dir="data", out_dir="results", scenarios=None, systems=None, use_stubs=False, strict=True):
    """Superset: jalankan seluruh pipeline pengujian dan simpan hasilnya."""
    scenarios = list(scenarios or DEFAULT_SCENARIOS)
    systems = list(systems or DEFAULT_SYSTEMS)
    backend = load_backend(use_stubs)
    timing = {}

    t = time.perf_counter()
    df = scan_dataset(data_dir)
    warnings = check_dataset(df, strict=strict)
    for w in warnings:
        print("[peringatan]", w)
    source = {s: "closed_speaker" if s == "loso" else s for s in scenarios}
    skipped = [s for s in scenarios if (df.scenario == source[s]).sum() == 0]
    scenarios = [s for s in scenarios if s not in skipped]
    needed = df[df.scenario.isin(["template"] + [source[s] for s in scenarios])]
    timing["scan_check_s"] = time.perf_counter() - t

    t = time.perf_counter()
    feats = extract_all(needed, backend.extract)
    timing["extract_s"] = time.perf_counter() - t

    t = time.perf_counter()
    trained = build_systems(df[df.scenario == "template"], feats, backend, systems)
    timing["train_s"] = time.perf_counter() - t

    results = []
    for scenario in scenarios:
        t = time.perf_counter()
        results.append(run_scenario(scenario, df, feats, trained, backend))
        timing[f"{scenario}_s"] = time.perf_counter() - t
        print(f"[selesai] {scenario}: {len(results[-1])} prediksi ({timing[f'{scenario}_s']:.1f} s)")
    results = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "predictions.csv", index=False)
    run_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backend": backend.name,
        "data_dir": str(data_dir),
        "scenarios": scenarios,
        "skipped_scenarios": skipped,
        "systems": systems,
        "n_files": {s: int((df.scenario == s).sum()) for s in SCENARIO_DIRS},
        "n_predictions": len(results),
        "warnings": warnings,
        "timing_s": {k: round(v, 2) for k, v in timing.items()},
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    if not results.empty:
        quick = results.pivot_table(index="scenario", columns="system", values="correct", aggfunc="mean") * 100
        print(f"\nCek cepat akurasi (%), backend={backend.name}; tabel resmi dihitung P5 dari predictions.csv:")
        print(quick.reindex([s for s in TEST_SCENARIOS if s in quick.index]).round(1).to_string())
    print(f"\nHasil disimpan di {out / 'predictions.csv'} dan {out / 'run_info.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Jalankan skenario pengujian pengenalan vowel.")
    parser.add_argument("--data", default="data", help="folder dataset (default: data)")
    parser.add_argument("--out", default="results", help="folder output (default: results)")
    parser.add_argument("--scenarios", nargs="+", choices=TEST_SCENARIOS, default=DEFAULT_SCENARIOS)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=DEFAULT_SYSTEMS)
    parser.add_argument("--use-stubs", action="store_true", help="pakai implementasi sementara src/_stubs.py")
    parser.add_argument("--no-strict", action="store_true", help="error pengecekan data jadi peringatan saja")
    args = parser.parse_args()
    run_all(args.data, args.out, args.scenarios, args.systems, args.use_stubs, strict=not args.no_strict)


if __name__ == "__main__":
    main()
