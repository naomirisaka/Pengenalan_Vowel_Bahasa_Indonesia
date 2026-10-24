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

from src.dtw import dtw_distance
from src.features import extract_mfcc_39
from src.gaussian import fit_gaussian, nll
from src.preprocessing import load_audio
from src.recognize import recognize_dtw, recognize_dtw_nn, recognize_gaussian
from src.template import build_averaged_template

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


FILENAME_RE = re.compile(r"^(?P<speaker>[A-Za-z0-9]+)_(?P<vowel>[AIUEO])_(?P<usage>CR|CS|OS)_(?P<condition>[A-Za-z]+)\.wav$", re.IGNORECASE)
EXPECTED_SAMPLE_RATE = 16000
MAX_DURATION = 10.0


WINLEN = 0.025
WINSTEP = 0.010
NUM_CEP = 13
N_FFT = 512


# Input : lokasi satu file audio
# Output: fitur MFCC 39 dimensi per frame (setelah preprocessing)
def extract_mfcc39(path):
    signal, sample_rate = load_audio(path, EXPECTED_SAMPLE_RATE)
    return extract_mfcc_39(signal, sample_rate, WINLEN, WINSTEP, NUM_CEP, N_FFT)


# Input : -
# Output: kumpulan fungsi ekstraksi fitur, pembuatan template, DTW, dan Gaussian
def load_backend():
    return SimpleNamespace(extract=extract_mfcc39, build_template=build_averaged_template,
                           dtw=dtw_distance, fit_gaussian=fit_gaussian, nll=nll)


# Input : folder dataset
# Output: tabel semua file audio beserta skenario, speaker, vowel, kondisi, dan info audionya
def scan_dataset(data_dir="data"):
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


# Input : tabel dataset, mode ketat (error menghentikan eksekusi atau tidak)
# Output: daftar peringatan; berhenti dengan error jika data tidak valid
def check_dataset(df, strict=True):
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

    speakers_per_vowel = {v: set(template[template.vowel == v].speaker) for v in VOWELS}
    for v, spk in speakers_per_vowel.items():
        if not spk:
            errors.append(f"Tidak ada data template untuk vowel {v}")
        elif len(spk) != 5:
            warnings.append(f"Template vowel {v} berisi {len(spk)} speaker (spek: 5)")
    if len({frozenset(s) for s in speakers_per_vowel.values() if s}) > 1:
        errors.append(f"Speaker template tidak sama di setiap vowel: { {v: sorted(s) for v, s in speakers_per_vowel.items()} }")
    template_speakers = set(template.speaker.dropna())

    for md5, group in named.groupby("md5"):
        if group.vowel.nunique() > 1:
            errors.append(f"File identik dengan label vowel berbeda: {sorted(group.path)}")
        for scenario, g in group.groupby("scenario"):
            if len(g) > 1:
                errors.append(f"File identik muncul {len(g)}x di {scenario}: {sorted(g.path)}")

    key = ["speaker", "vowel", "md5"]
    if set(map(tuple, by["closed_recording"][key].values)) != set(map(tuple, template[key].values)):
        errors.append("Isi test/closed/recording tidak identik dengan data/template")

    leaked = by["closed_speaker"][by["closed_speaker"].md5.isin(set(template.md5))]
    for f in leaked.path:
        errors.append(f"Data closed speaker identik dengan rekaman template: {f}")
    for spk in sorted(set(by["closed_speaker"].speaker.dropna()) - template_speakers):
        errors.append(f"Speaker {spk} di closed speaker tidak ada di template")

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


# Input : tabel file audio, fungsi ekstraksi fitur
# Output: fitur MFCC setiap file (file identik hanya diekstrak sekali)
def extract_all(df, extract_fn):
    by_md5 = {}
    for path, md5 in zip(df.path, df.md5):
        if md5 not in by_md5:
            by_md5[md5] = extract_fn(path)
    return {path: by_md5[md5] for path, md5 in zip(df.path, df.md5)}


# Input : data latih (rekaman template), fitur MFCC-nya, fungsi P1-P3, sistem yang dipakai
# Output: model tiap vowel untuk setiap sistem (averaged template DTW, model Gaussian)
def build_systems(train_df, feats, backend, systems=DEFAULT_SYSTEMS):
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


# Input : data uji, fitur MFCC-nya, model tiap sistem, fungsi P1-P3, nama skenario
# Output: hasil pengenalan setiap rekaman uji oleh setiap sistem (label asli, tebakan, skor)
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


# Input : nama skenario uji, tabel dataset, fitur MFCC, model tiap sistem, fungsi P1-P3
# Output: tabel hasil pengenalan untuk skenario tersebut
def run_scenario(scenario, df, feats, trained, backend):
    if scenario == "loso":
        template, closed = df[df.scenario == "template"], df[df.scenario == "closed_speaker"]
        rows = []
        for spk in sorted(template.speaker.unique()):
            fold = build_systems(template[template.speaker != spk], feats, backend, list(trained))
            rows += _evaluate(closed[closed.speaker == spk], feats, fold, backend, scenario)
    else:
        rows = _evaluate(df[df.scenario == scenario], feats, trained, backend, scenario)
    return pd.DataFrame(rows)


# Input : folder dataset, folder hasil, skenario dan sistem yang dijalankan, mode ketat
# Output: tabel hasil pengenalan semua skenario, disimpan ke predictions.csv dan run_info.json
def run_all(data_dir="data", out_dir="results", scenarios=None, systems=None, strict=True):
    scenarios = list(scenarios or DEFAULT_SCENARIOS)
    systems = list(systems or DEFAULT_SYSTEMS)
    backend = load_backend()
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
        "data_dir": str(data_dir),
        "mfcc": {"winlen": WINLEN, "winstep": WINSTEP, "num_cep": NUM_CEP, "n_fft": N_FFT},
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
        print("\nCek cepat akurasi (%); tabel resmi dihitung P5 dari predictions.csv:")
        print(quick.reindex([s for s in TEST_SCENARIOS if s in quick.index]).round(1).to_string())
    print(f"\nHasil disimpan di {out / 'predictions.csv'} dan {out / 'run_info.json'}")
    return results


# Input : pilihan dari command line (folder data, folder hasil, skenario, sistem)
# Output: menjalankan seluruh pengujian
def main():
    parser = argparse.ArgumentParser(description="Jalankan skenario pengujian pengenalan vowel.")
    parser.add_argument("--data", default="data", help="folder dataset (default: data)")
    parser.add_argument("--out", default="results", help="folder output (default: results)")
    parser.add_argument("--scenarios", nargs="+", choices=TEST_SCENARIOS, default=DEFAULT_SCENARIOS)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=DEFAULT_SYSTEMS)
    parser.add_argument("--no-strict", action="store_true", help="error pengecekan data jadi peringatan saja")
    args = parser.parse_args()
    run_all(args.data, args.out, args.scenarios, args.systems, strict=not args.no_strict)


if __name__ == "__main__":
    main()
