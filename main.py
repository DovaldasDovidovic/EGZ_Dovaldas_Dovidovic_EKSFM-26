"""Pagrindinio UCI HAR eksperimento atkūrimas viena komanda: python main.py

Nenaudoja results/subject_disjoint_split.npz, Jupyter būsenos ir išsaugotų modelių.
Hiperparametrai fiksuoti iš 2–3 etapų TRAIN GroupKFold (čia GridSearchCV nevykdomas).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REFERENCE_CSV = RESULTS_DIR / "final_model_comparison.csv"
REPRODUCED_CSV = RESULTS_DIR / "reproduced_main_results.csv"

EXPECTED = {
    "n_samples": 10299,
    "n_features": 561,
    "n_subjects": 30,
    "n_train": 7144,
    "n_test": 3155,
    "n_train_subjects": 21,
    "n_test_subjects": 9,
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def find_uci_har_root(data_root: Path) -> Path:
    if not data_root.exists():
        fail(
            "Nerastas katalogas data/. Įdėkite oficialų UCI Human Activity "
            "Recognition Using Smartphones rinkinį į data/ pagal README."
        )
    matches = []
    for features_path in data_root.rglob("features.txt"):
        if "__MACOSX" in str(features_path):
            continue
        root = features_path.parent
        if (
            (root / "activity_labels.txt").is_file()
            and (root / "train").is_dir()
            and (root / "test").is_dir()
            and (root / "train" / "X_train.txt").is_file()
            and (root / "train" / "y_train.txt").is_file()
            and (root / "train" / "subject_train.txt").is_file()
            and (root / "test" / "X_test.txt").is_file()
            and (root / "test" / "y_test.txt").is_file()
            and (root / "test" / "subject_test.txt").is_file()
        ):
            matches.append(root)
    unique = []
    seen = set()
    for path in matches:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        fail(
            "Nerastas UCI HAR Dataset po data/. Reikia features.txt, "
            "activity_labels.txt, train/ ir test/ (su X_*.txt, y_*.txt, subject_*.txt). "
            "Žr. README skyrių Reproducibility / How to run."
        )
    unique.sort(key=lambda p: len(p.parts))
    return unique[0]


def make_unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique.append(name if count == 1 else f"{name}_{count}")
    return unique


def load_feature_table(path: Path, feature_names: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=feature_names,
        engine="python",
    )


def load_vector(path: Path, column_name: str) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", header=None, names=[column_name], engine="python"
    )


def load_uci_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features_meta = pd.read_csv(
        data_dir / "features.txt",
        sep=r"\s+",
        header=None,
        names=["feature_id", "feature_name"],
        engine="python",
    )
    feature_names = make_unique_names(features_meta["feature_name"].tolist())
    x_train = load_feature_table(data_dir / "train" / "X_train.txt", feature_names)
    y_train = load_vector(data_dir / "train" / "y_train.txt", "activity_id")
    subject_train = load_vector(
        data_dir / "train" / "subject_train.txt", "subject_id"
    )
    x_test = load_feature_table(data_dir / "test" / "X_test.txt", feature_names)
    y_test = load_vector(data_dir / "test" / "y_test.txt", "activity_id")
    subject_test = load_vector(data_dir / "test" / "subject_test.txt", "subject_id")

    x = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    y = pd.concat([y_train, y_test], axis=0, ignore_index=True)["activity_id"]
    subjects = pd.concat([subject_train, subject_test], axis=0, ignore_index=True)[
        "subject_id"
    ]
    return x, y, subjects


def make_subject_disjoint_split(
    x: pd.DataFrame, y: pd.Series, subjects: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(x, y, groups=subjects))
    x_train = x.iloc[train_idx].reset_index(drop=True)
    x_test = x.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    subjects_train = subjects.iloc[train_idx].reset_index(drop=True)
    subjects_test = subjects.iloc[test_idx].reset_index(drop=True)
    return x_train, x_test, y_train, y_test, subjects_train, subjects_test


def assert_protocol(
    x: pd.DataFrame,
    y: pd.Series,
    subjects: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    subjects_train: pd.Series,
    subjects_test: pd.Series,
) -> None:
    checks = {
        "n_samples": len(x),
        "n_features": x.shape[1],
        "n_subjects": int(subjects.nunique()),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "n_train_subjects": int(subjects_train.nunique()),
        "n_test_subjects": int(subjects_test.nunique()),
    }
    for key, expected in EXPECTED.items():
        got = checks[key]
        if got != expected:
            fail(f"Protokolo patikra nepavyko: {key}={got}, tikėtasi {expected}.")
    overlap = set(subjects_train.tolist()) & set(subjects_test.tolist())
    if overlap:
        fail(f"TRAIN ir TEST subject sankirta turi būti tuščia, gauta: {sorted(overlap)}")
    if len(y) != len(x) or len(subjects) != len(x):
        fail("X, y ir subject eilučių skaičius nesutampa.")


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Model": name,
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro")),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "TEST Errors": int(np.sum(y_pred != y_true)),
    }


def build_models() -> dict[str, Pipeline]:
    return {
        "k-NN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(n_neighbors=11, weights="distance"),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        C=0.01,
                        random_state=RANDOM_STATE,
                        max_iter=10000,
                        dual="auto",
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(100,),
                        alpha=0.001,
                        random_state=RANDOM_STATE,
                        max_iter=500,
                    ),
                ),
            ]
        ),
    }


def check_reproducibility(results: pd.DataFrame) -> None:
    if not REFERENCE_CSV.exists():
        print(
            f"REPRODUCIBILITY CHECK: SKIPPED ({REFERENCE_CSV.name} nerastas)."
        )
        return
    reference = pd.read_csv(REFERENCE_CSV)
    ref_errors_col = (
        "TEST Errors"
        if "TEST Errors" in reference.columns
        else "Number of TEST Errors"
    )
    mismatches = []
    for _, row in results.iterrows():
        model = row["Model"]
        ref = reference.loc[reference["Model"] == model]
        if ref.empty:
            mismatches.append(f"{model}: nėra eilutės {REFERENCE_CSV.name}")
            continue
        ref = ref.iloc[0]
        for metric in ["Macro-F1", "Balanced Accuracy"]:
            if not np.isclose(
                row[metric], ref[metric], rtol=0.0, atol=1e-10
            ):
                mismatches.append(
                    f"{model} {metric}: gauta {row[metric]!r}, "
                    f"tikėtasi {float(ref[metric])!r}"
                )
        got_err = int(row["TEST Errors"])
        exp_err = int(ref[ref_errors_col])
        if got_err != exp_err:
            mismatches.append(
                f"{model} TEST Errors: gauta {got_err}, tikėtasi {exp_err}"
            )
    if mismatches:
        details = "\n".join(f"  - {item}" for item in mismatches)
        fail("REPRODUCIBILITY CHECK: FAILED\n" + details)
    print("REPRODUCIBILITY CHECK: PASSED")


def main() -> None:
    np.random.seed(RANDOM_STATE)
    data_dir = find_uci_har_root(DATA_ROOT)
    print(f"RANDOM_STATE = {RANDOM_STATE}")
    print(f"Dataset path: {data_dir}")

    x, y, subjects = load_uci_tables(data_dir)
    print(f"Combined dataset: {x.shape[0]} samples, {x.shape[1]} features")
    print(f"Unique subjects (all): {sorted(subjects.unique().tolist())}")

    x_train, x_test, y_train, y_test, subjects_train, subjects_test = (
        make_subject_disjoint_split(x, y, subjects)
    )
    assert_protocol(x, y, subjects, x_train, x_test, subjects_train, subjects_test)

    train_ids = sorted(subjects_train.unique().tolist())
    test_ids = sorted(subjects_test.unique().tolist())
    overlap = sorted(set(train_ids) & set(test_ids))
    print(f"TRAIN: {x_train.shape[0]} samples, {len(train_ids)} subjects")
    print(f"TEST:  {x_test.shape[0]} samples, {len(test_ids)} subjects")
    print(f"TRAIN subject IDs: {train_ids}")
    print(f"TEST subject IDs:  {test_ids}")
    print(f"Subject overlap: {overlap}")

    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    x_train_np = x_train.to_numpy(dtype=np.float64)
    x_test_np = x_test.to_numpy(dtype=np.float64)

    values, counts = np.unique(y_train_np, return_counts=True)
    majority_class = int(values[np.argmax(counts)])
    y_pred_majority = np.full_like(y_test_np, majority_class)
    rows = [evaluate("Majority Class", y_test_np, y_pred_majority)]
    print(f"Majority class (TRAIN only): {majority_class}")

    for name, pipe in build_models().items():
        pipe.fit(x_train_np, y_train_np)
        y_pred = pipe.predict(x_test_np)
        rows.append(evaluate(name, y_test_np, y_pred))

    results = pd.DataFrame(rows)
    print()
    print(results.to_string(index=False))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPRODUCED_CSV, index=False)
    print(f"Saved: {REPRODUCED_CSV}")
    check_reproducibility(results)


if __name__ == "__main__":
    main()
