from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "news_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


def main():

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    required = {"text", "label"}

    if not required.issubset(df.columns):
        raise ValueError("CSV must contain text and label columns.")

    # 2. Clean dataset
    df = df.dropna(subset=["text", "label"]).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.upper().str.strip()

    df = df[df["label"].isin(["REAL", "FAKE"])]
    df = df[df["text"].str.len() >= 10]
    df = df.drop_duplicates(subset=["text"])

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain REAL and FAKE labels.")

    if df["label"].value_counts().min() < 5:
        raise ValueError(
            "At least 5 examples of each label are required."
        )

    print("\nDATASET INFORMATION")
    print("-------------------")
    print(df["label"].value_counts())

    # 3. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"]
    )

    # 4. Word and Character TF-IDF
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=12000,
        sublinear_tf=True
    )

    char_tfidf = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=12000,
        sublinear_tf=True
    )

    # 5. Combine both feature extractors
    features = FeatureUnion([
        ("word_features", word_tfidf),
        ("character_features", char_tfidf)
    ])

    # 6. Create ML pipeline
    pipeline = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 7. Cross-validation on training data
    min_class_count = int(y_train.value_counts().min())
    folds = min(5, min_class_count)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=folds,
        scoring="accuracy"
    )

    # 8. Train final model
    pipeline.fit(X_train, y_train)

    # 9. Test model
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    # 10. Save evaluation results
    metrics = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "label_distribution": {
            str(k): int(v)
            for k, v in df["label"].value_counts().items()
        },
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "cross_validation_mean": round(float(cv_scores.mean()), 4),
        "cross_validation_std": round(float(cv_scores.std()), 4),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=["FAKE", "REAL"]
        ).tolist(),
        "labels_order": ["FAKE", "REAL"]
    }

    # 11. Save trained model
    joblib.dump(
        pipeline,
        MODEL_DIR / "fake_news_pipeline.joblib"
    )

    with open(
        MODEL_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    # 12. Display results
    print("\nMODEL TRAINING COMPLETED")
    print("------------------------")

    print(f"Dataset rows : {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    print(f"\nAccuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1-score  : {f1:.3f}")

    print("\nCROSS VALIDATION")
    print("----------------")
    print(f"Folds       : {folds}")
    print(f"Mean        : {cv_scores.mean() * 100:.2f}%")
    print(f"Std. Dev.   : {cv_scores.std() * 100:.2f}%")

    print("\nCONFUSION MATRIX")
    print("----------------")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["FAKE", "REAL"]
    ))

    print("\nSaved model and metrics successfully.")

    print(
        "\nNOTE: Results from a small educational dataset "
        "do not represent real-world fact-checking accuracy."
    )


if __name__ == "__main__":
    main()