"""Baseline analysis for the waste classification dataset.
 
Usage:
    python baseline.py                      # uses Classifying_waste_dataset.csv next to this file
    python baseline.py --save-table         # also writes the model comparison to table.csv
    python baseline.py --data path/to.csv   # use a CSV somewhere else
 
Requires: pandas, scikit-learn (pip install -r requirements.txt)
"""
import argparse
import sys
from pathlib import Path
 
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
 
FEATURES = ["Weight", "Color", "Texture", "Odor"]
LABEL = "Type"
DEFAULT_CSV = Path(__file__).resolve().parent / "Classifying_waste_dataset.csv"
 
 
def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV and make sure it has the columns we need."""
    if not path.is_file():
        sys.exit(
            f"Could not find the dataset at: {path}\n"
            "Keep Classifying_waste_dataset.csv in the same folder as baseline.py, "
            "or pass its location with --data."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        sys.exit(f"The CSV is missing these columns: {', '.join(missing)}")
    return df
 
 
def describe(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print("\nClass counts:")
    print(df[LABEL].value_counts().to_string())
    print("\nClass share:")
    print(df[LABEL].value_counts(normalize=True).round(3).to_string())
    print("\nAverage feature values by class:")
    print(df.groupby(LABEL)[FEATURES].mean().round(3).to_string())
 
 
def cross_validate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    models = {
        "Majority class (always Organic)": DummyClassifier(strategy="most_frequent"),
        "Decision tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=0),
        "Random forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=0),
    }
    print("\n5-fold cross-validation:")
    print(f"  {'Model':<34}{'Accuracy':>10}{'Macro F1':>10}")
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1_macro"])
        accuracy = scores["test_accuracy"].mean()
        macro_f1 = scores["test_f1_macro"].mean()
        print(f"  {name:<34}{accuracy:>10.3f}{macro_f1:>10.3f}")
        rows.append({"Model": name, "Accuracy": round(accuracy, 3), "Macro_F1": round(macro_f1, 3)})
    return pd.DataFrame(rows)
 
 
def holdout_report(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    forest = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
    forest.fit(X_train, y_train)
    predictions = forest.predict(X_test)
 
    print("\nHold-out test set (20%), random forest with balanced class weights:")
    print(classification_report(y_test, predictions, zero_division=0))
 
    labels = sorted(y.unique())
    matrix = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=[f"actual {c}" for c in labels],
        columns=[f"predicted {c}" for c in labels],
    )
    print("Confusion matrix:")
    print(matrix.to_string())
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline analysis for the waste dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_CSV, help="path to the CSV file")
    parser.add_argument(
        "--save-table",
        action="store_true",
        help="also write the model comparison to table.csv next to this script",
    )
    args = parser.parse_args()
 
    df = load_data(args.data)
    X, y = df[FEATURES], df[LABEL]
 
    describe(df)
    results = cross_validate_models(X, y)
    holdout_report(X, y)
 
    if args.save_table:
        out_path = Path(__file__).resolve().parent / "table.csv"
        results.to_csv(out_path, index=False)
        print(f"\nSaved model comparison to {out_path}")
 
 
if __name__ == "__main__":
    main()
 
