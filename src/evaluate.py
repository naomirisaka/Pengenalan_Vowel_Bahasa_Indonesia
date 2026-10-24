"""Evaluasi akurasi dan confusion matrix pengenalan vowel.

Menghasilkan:
- Ringkasan : Akurasi per sistem x skenario
- Tabel 1   : Akurasi per vowel
- Tabel 2   : Closed experiments per speaker
- Tabel 3   : Open experiments per speaker
- Confusion Matrix per sistem dan skenario
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

VOWELS = ["A", "I", "U", "E", "O"]
SCENARIOS = ["closed_recording", "closed_speaker", "open", "loso"]
SYSTEMS = ["dtw", "gaussian", "dtw_nn"]

SCENARIO_LABELS = {
    "closed_recording": "Closed Recording",
    "closed_speaker": "Closed Speakers",
    "open": "Open Speakers",
    "loso": "LOSO",
}
SYSTEM_LABELS = {
    "dtw": "DTW Template",
    "gaussian": "Model Gaussian",
    "dtw_nn": "DTW NN",
}
AVERAGE_SCENARIOS = ["closed_recording", "closed_speaker", "open"]

CONDITIONS = ["N", "HP", "LP", "SD", "LD"]
CONDITION_LABELS = {
    "N": "Normal",
    "HP": "High Pitch",
    "LP": "Low Pitch",
    "SD": "Short Duration",
    "LD": "Long Duration",
}
CONDITION_GROUPS = {
    "normal": ["N"],
    "pitch": ["HP", "LP"],
    "duration": ["SD", "LD"],
    "all": None,
}
CONDITION_GROUP_LABELS = {
    "normal": "Normal",
    "pitch": "Variasi Pitch",
    "duration": "Variasi Durasi",
    "all": "Overall",
}

def load_predictions(path: str | Path = "results/predictions.csv") -> pd.DataFrame:
    """Baca dan validasi predictions.csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File prediksi tidak ditemukan: {path}. "
            "Jalankan `python -m src.run_experiments` terlebih dahulu."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"scenario", "system", "vowel_true", "pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di {path}: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"File prediksi kosong: {path}")

    df = df.copy()
    df["scenario"] = df["scenario"].astype(str).str.strip().str.lower()
    df["system"] = df["system"].astype(str).str.strip().str.lower()
    for col in ("vowel_true", "pred"):
        df[col] = df[col].astype(str).str.strip().str.upper()
    for col in ("speaker", "file", "condition"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "condition" in df.columns:
        df["condition"] = df["condition"].str.upper()

    unknown = (set(df["vowel_true"]) | set(df["pred"])) - set(VOWELS)
    if unknown:
        raise ValueError(f"Label vowel tidak dikenal: {sorted(unknown)}")

    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(int)
    else:
        df["correct"] = (df["vowel_true"] == df["pred"]).astype(int)

    if "file" in df.columns:
        duplicate = df.duplicated(subset=["scenario", "system", "file"])
        if duplicate.any():
            contoh = df.loc[duplicate, ["scenario", "system", "file"]].head(3).values.tolist()
            raise ValueError(f"Prediksi duplikat untuk file yang sama: {contoh}")

    return df


# --- Helper ---

def _sorted_systems(values: Iterable[str]) -> list[str]:
    present = set(values)
    return [s for s in SYSTEMS if s in present] + sorted(present - set(SYSTEMS))


def _sorted_scenarios(values: Iterable[str]) -> list[str]:
    present = set(values)
    return [s for s in SCENARIOS if s in present] + sorted(present - set(SCENARIOS))


def _average_scenarios(columns: Iterable[str]) -> list[str]:
    columns = list(columns)
    utama = [s for s in AVERAGE_SCENARIOS if s in columns]
    return utama if utama else [s for s in columns if s != "loso"]


def _rename_axis(df: pd.DataFrame, labels: bool) -> pd.DataFrame:
    if not labels:
        return df
    df = df.copy()
    df.index = df.index.map(lambda x: SYSTEM_LABELS.get(x, x))
    df.columns = [SCENARIO_LABELS.get(c, c) for c in df.columns]
    df.index.name = "Sistem"
    return df


def _system_label(value: str, labels: bool) -> str:
    return SYSTEM_LABELS.get(value, value) if labels else value


def _scenario_label(value: str, labels: bool) -> str:
    return SCENARIO_LABELS.get(value, value) if labels else value


def filter_condition(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Ambil prediksi untuk satu kelompok kondisi (normal, pitch, duration, all)."""
    if group not in CONDITION_GROUPS:
        raise ValueError(
            f"Kelompok kondisi tidak dikenal: {group!r}. Pilih dari {list(CONDITION_GROUPS)}"
        )
    codes = CONDITION_GROUPS[group]
    if codes is None:
        return df
    if "condition" not in df.columns:
        raise ValueError("Kolom `condition` tidak ada di predictions.csv")
    return df[df["condition"].isin(codes)]


def _combined_label(system: str, scenario: str, labels: bool = True) -> str:
    khusus = {
        ("dtw", "closed_recording"): "DTW Closed Recording",
        ("dtw", "closed_speaker"): "DTW Closed Speakers",
        ("dtw", "open"): "DTW Open Speakers",
        ("gaussian", "closed_recording"): "Gaussian Closed Recordings",
        ("gaussian", "closed_speaker"): "Gaussian Closed Speakers",
        ("gaussian", "open"): "Gaussian Open",
    }
    if labels and (system, scenario) in khusus:
        return khusus[(system, scenario)]
    sistem = SYSTEM_LABELS.get(system, system) if labels else system
    skenario = SCENARIO_LABELS.get(scenario, scenario) if labels else scenario
    return f"{sistem} {skenario}"


# --- Ringkasan Akurasi ---

def overall_accuracy(df: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
    """Ringkasan akurasi (%) per sistem x skenario."""
    scenarios = _sorted_scenarios(df["scenario"].unique())
    table = (
        df.pivot_table(index="system", columns="scenario", values="correct", aggfunc="mean")
        .reindex(index=_sorted_systems(df["system"].unique()), columns=scenarios)
        * 100
    )
    rata_rata = _average_scenarios(scenarios)
    table["average"] = table[rata_rata].mean(axis=1) if rata_rata else np.nan
    table = table.rename(columns={"average": "Rata-rata"})
    return _rename_axis(table, labels)


# --- Akurasi per Kondisi (Normal vs Variasi) ---

def accuracy_per_condition(df: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
    """Akurasi (%) per sistem x skenario untuk tiap kondisi dan kelompok kondisi.    """
    if "condition" not in df.columns:
        raise ValueError("Kolom `condition` tidak ada di predictions.csv")

    per_kode = df.pivot_table(
        index=["system", "scenario"], columns="condition", values="correct", aggfunc="mean"
    )
    kode = [c for c in CONDITIONS if c in per_kode.columns]
    kode += sorted(set(per_kode.columns) - set(CONDITIONS))
    table = per_kode[kode].copy()

    for group, codes in CONDITION_GROUPS.items():
        if codes is not None and len(codes) == 1:
            continue  # sama dengan kolom kode kondisinya
        subset = filter_condition(df, group)
        if subset.empty:
            continue
        table[group] = subset.groupby(["system", "scenario"])["correct"].mean()

    order = [
        (s, sc)
        for s in _sorted_systems(df["system"].unique())
        for sc in _sorted_scenarios(df["scenario"].unique())
        if (s, sc) in table.index
    ]
    table = table.reindex(pd.MultiIndex.from_tuples(order, names=["system", "scenario"])) * 100

    if labels:
        table.index = table.index.map(
            lambda x: (_system_label(x[0], True), _scenario_label(x[1], True))
        )
        table.index.names = ["Sistem", "Skenario"]
        table.columns = [
            CONDITION_LABELS.get(c, CONDITION_GROUP_LABELS.get(c, c)) for c in table.columns
        ]
    return table


# --- Tabel 1: Akurasi per Vowel ---

def accuracy_per_vowel(df: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
    """Tabel 1: Akurasi (%) per vowel beserta Average Accuracy."""
    grouped = (
        df.groupby(["vowel_true", "system", "scenario"])["correct"]
        .agg(["count", "mean"])
        .reset_index()
    )
    grouped["accuracy"] = grouped["mean"] * 100

    wide = grouped.pivot_table(
        index="vowel_true", columns=["system", "scenario"], values="accuracy"
    )
    systems = _sorted_systems(df["system"].unique())
    scenarios = _sorted_scenarios(df["scenario"].unique())
    columns = [(s, sc) for s in systems for sc in scenarios if (s, sc) in wide.columns]
    wide = wide.reindex(index=VOWELS, columns=pd.MultiIndex.from_tuples(columns))
    wide.loc["Average Accuracy"] = wide.mean(axis=0, skipna=True)

    if labels:
        wide.columns = [_combined_label(s, sc, True) for s, sc in wide.columns]
        wide.index.name = "Vowel"
    return wide


def accuracy_per_vowel_detail(df: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
    """Detail akurasi (%) per sistem, skenario, dan vowel."""
    grouped = (
        df.groupby(["system", "scenario", "vowel_true"])["correct"]
        .agg(n="count", accuracy="mean")
        .reset_index()
    )
    grouped["accuracy"] *= 100
    grouped = grouped.sort_values(["system", "scenario", "vowel_true"])
    if labels:
        grouped["system"] = grouped["system"].map(lambda x: _system_label(x, True))
        grouped["scenario"] = grouped["scenario"].map(lambda x: _scenario_label(x, True))
    return grouped.rename(
        columns={"system": "Sistem", "scenario": "Skenario", "vowel_true": "Vowel",
                 "accuracy": "Akurasi (%)"}
    ).reset_index(drop=True)


# --- Tabel 2 & 3: Akurasi per Speaker ---
def accuracy_per_speaker(df: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
    """Detail data dan akurasi (%) per speaker, skenario, dan sistem."""
    if "speaker" not in df.columns:
        raise ValueError("Kolom `speaker` tidak ada di predictions.csv")

    grouped = (
        df.groupby(["system", "scenario", "speaker"])["correct"]
        .agg(n="count", accuracy="mean")
        .reset_index()
    )
    grouped["accuracy"] *= 100
    grouped = grouped.sort_values(["system", "scenario", "speaker"])
    if labels:
        grouped["system"] = grouped["system"].map(lambda x: _system_label(x, True))
        grouped["scenario"] = grouped["scenario"].map(lambda x: _scenario_label(x, True))
    return grouped.rename(
        columns={"system": "Sistem", "scenario": "Skenario", "speaker": "Speaker",
                 "accuracy": "Akurasi (%)"}
    ).reset_index(drop=True)


def _speaker_matrix(
    df: pd.DataFrame,
    scenarios: Iterable[str],
    labels: bool = True,
    is_open: bool = False,
) -> pd.DataFrame:
    """Matriks akurasi per speaker dengan baris dan kolom Average."""
    scenarios = list(scenarios)
    subset = df[df["scenario"].isin(scenarios)]
    if subset.empty:
        return pd.DataFrame()

    grouped = (
        subset.groupby(["speaker", "system", "scenario"])["correct"].mean() * 100
    )
    wide = grouped.unstack(["system", "scenario"])
    systems = _sorted_systems(subset["system"].unique())
    columns = [(s, sc) for s in systems for sc in scenarios if (s, sc) in wide.columns]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(columns))

    # Kolom dan baris Average
    wide["Average"] = wide.mean(axis=1)
    wide.loc["Average"] = wide.mean(axis=0, skipna=True)

    wide.columns = [
        "Average" if isinstance(col, tuple) and col[0] == "Average" else col
        for col in wide.columns
    ]

    if labels:
        def _get_col_label(col):
            if col == "Average":
                return "Average"
            system, scenario = col
            if is_open:
                if system == "dtw" and scenario == "open":
                    return "DTW Open"
                if system == "gaussian" and scenario == "open":
                    return "Gaussian Open"
            return _combined_label(system, scenario, True)

        wide.columns = [_get_col_label(col) for col in wide.columns]
        wide.index.name = "Speakers"
    return wide


def accuracy_per_speaker_closed(df: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
    """Tabel 2: Closed Experiments per Speaker (%)."""
    return _speaker_matrix(
        df, ["closed_recording", "closed_speaker"], labels=labels, is_open=False
    )


def accuracy_per_speaker_open(df: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
    """Tabel 3: Open Experiments per Speaker (%)."""
    return _speaker_matrix(
        df, ["open"], labels=labels, is_open=True
    )


def accuracy_per_speaker_pivot(df: pd.DataFrame, labels: bool = True) -> pd.DataFrame:
    """Matriks pivot akurasi: baris speaker, kolom (skenario, sistem)."""
    long = accuracy_per_speaker(df, labels=labels)
    return long.pivot_table(
        index="Speaker", columns=["Skenario", "Sistem"], values="Akurasi (%)"
    )


# --- Confusion Matrix ---

def confusion_matrix(
    df: pd.DataFrame,
    system: str,
    scenario: Optional[str] = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Hitung confusion matrix vowel untuk sistem dan skenario tertentu."""
    subset = df[df["system"] == system]
    if scenario is not None:
        subset = subset[subset["scenario"] == scenario]
    if subset.empty:
        raise ValueError(f"Tidak ada prediksi untuk system={system!r}, scenario={scenario!r}")

    cm = pd.crosstab(subset["vowel_true"], subset["pred"])
    cm = cm.reindex(index=VOWELS, columns=VOWELS, fill_value=0)
    cm.index.name = "Aktual"
    cm.columns.name = "Prediksi"
    if normalize:
        total = cm.sum(axis=1).replace(0, np.nan)
        cm = (cm.div(total, axis=0) * 100).fillna(0.0)
    return cm


def all_confusion_matrices(df: pd.DataFrame, normalize: bool = False) -> Dict[tuple[str, str], pd.DataFrame]:
    """Confusion matrix untuk setiap kombinasi sistem x skenario."""
    return {
        (system, scenario): confusion_matrix(df, system, scenario, normalize=normalize)
        for system in _sorted_systems(df["system"].unique())
        for scenario in _sorted_scenarios(df["scenario"].unique())
        if not df[(df["system"] == system) & (df["scenario"] == scenario)].empty
    }


# --- Export Evaluasi ---

def _evaluate_subset(df: pd.DataFrame, out: Optional[Path]) -> Dict[str, object]:
    """Hitung semua tabel untuk satu subset prediksi; simpan ke `out` bila diberikan."""
    ringkasan = overall_accuracy(df, labels=True)
    table1 = accuracy_per_vowel(df, labels=True)
    table1_detail = accuracy_per_vowel_detail(df, labels=True)
    table2 = accuracy_per_speaker_closed(df, labels=True)
    table3 = accuracy_per_speaker_open(df, labels=True)
    table_detail = accuracy_per_speaker(df, labels=True)
    matrices = all_confusion_matrices(df)

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        ringkasan.to_csv(out / "table_overall.csv")
        table1.to_csv(out / "table1_per_vowel.csv")
        table1_detail.to_csv(out / "table1_per_vowel_detail.csv", index=False)
        if not table2.empty:
            table2.to_csv(out / "table2_closed_speakers.csv")
        if not table3.empty:
            table3.to_csv(out / "table3_open_speakers.csv")
        table_detail.to_csv(out / "table_detail_per_speaker.csv", index=False)
        accuracy_per_speaker_pivot(df, labels=True).to_csv(
            out / "table_detail_per_speaker_pivot.csv"
        )
        cm_dir = out / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        for (system, scenario), cm in matrices.items():
            cm.to_csv(cm_dir / f"{system}_{scenario}.csv")
            (cm_dir / f"{system}_{scenario}_normalized.csv").write_text(
                confusion_matrix(df, system, scenario, normalize=True).round(2).to_csv(),
                encoding="utf-8",
            )

    return {
        "ringkasan": ringkasan,
        "table1": table1,
        "table1_detail": table1_detail,
        "table2": table2,
        "table3": table3,
        "table_detail": table_detail,
        "matrices": matrices,
    }


def evaluate_all(
    df: Optional[pd.DataFrame] = None,
    predictions_path: str | Path = "results/predictions.csv",
    out_dir: str | Path = "results/evaluation",
    save: bool = True,
) -> Dict[str, object]:
    """Hitung semua evaluasi, terpisah per kelompok kondisi, dan simpan ke CSV."""
    if df is None:
        df = load_predictions(predictions_path)
    if df.empty:
        raise ValueError("Tidak ada prediksi untuk dievaluasi")

    out = Path(out_dir) if save else None

    if "condition" not in df.columns:
        hasil = _evaluate_subset(df, out)
        hasil["kelompok"] = {"all": dict(hasil)}
        hasil["per_kondisi"] = pd.DataFrame()
        return hasil

    kelompok = {}
    for group in CONDITION_GROUPS:
        subset = filter_condition(df, group)
        if subset.empty:
            continue
        kelompok[group] = _evaluate_subset(subset, out / group if out else None)

    per_kondisi = accuracy_per_condition(df, labels=True)
    if out is not None:
        per_kondisi.to_csv(out / "table_per_condition.csv")

    utama = kelompok.get("normal") or kelompok["all"]
    return {**utama, "kelompok": kelompok, "per_kondisi": per_kondisi}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hitung akurasi dan confusion matrix.")
    parser.add_argument("--predictions", default="results/predictions.csv",
                        help="file predictions.csv (default: results/predictions.csv)")
    parser.add_argument("--out", default="results/evaluation",
                        help="folder output evaluasi (default: results/evaluation)")
    parser.add_argument("--no-save", action="store_true", help="jangan menulis file output")
    args = parser.parse_args()

    hasil = evaluate_all(
        predictions_path=args.predictions,
        out_dir=args.out,
        save=not args.no_save,
    )
    for group, sub in hasil["kelompok"].items():
        print(f"Ringkasan akurasi - {CONDITION_GROUP_LABELS.get(group, group)} (%)")
        print(sub["ringkasan"].round(2).to_string())
        print()
    if not hasil["per_kondisi"].empty:
        print("Akurasi per kondisi (%)")
        print(hasil["per_kondisi"].round(2).to_string())
        print()
    print("=== Tabel berikut untuk kondisi Normal ===")
    print("\nTabel 1. Akurasi per vowel (%)")
    print(hasil["table1"].round(2).to_string())
    print("\nTabel 2. Closed experiments per speaker (%)")
    print(hasil["table2"].round(2).to_string() if not hasil["table2"].empty
          else "Belum ada data.")
    print("\nTabel 3. Open experiments per speaker (%)")
    print(hasil["table3"].round(2).to_string() if not hasil["table3"].empty
          else "Belum ada data.")
    if not args.no_save:
        print(f"\nTabel CSV disimpan di {Path(args.out)}")


if __name__ == "__main__":
    main()
