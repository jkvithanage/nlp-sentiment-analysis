from pathlib import Path
import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from utils import (
    stratified_splits,
    make_tfidf,
    vectorize,
    report,
    save_artifacts,
)
from data_loader import load_data
from preprocess import preprocess_reviews


# Config
DATA_DIR = "data"
VAL_SIZE = 0.1
TEST_SIZE = 0.1
SEED = 42

# Vectorizer options
NGRAM_MAX = 2
MIN_DF = 2
MAX_DF = 0.9

NB_MAX_FEATURES = None  # keep full vocab for NB
LR_MAX_FEATURES = 100_000  # cap vocab for LR (None to disable)

# NB hyperparameters
NB_ALPHA = 0.5

# LR hyperparameters
LR_C = 1.0
LR_PENALTY = "l2"
LR_CLASS_WEIGHT = None
LR_SOLVER = "liblinear"


def _load_texts_labels():
    preprocessed_file = Path("preprocessed.csv")
    if preprocessed_file.exists():
        texts = []
        labels_list = []
        with preprocessed_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise SystemExit("preprocessed.csv missing required columns 'text,label'")
            for row in reader:
                texts.append(row["text"])
                labels_list.append(int(row["label"]))
        labels = np.array(labels_list, dtype=int)
        print(f"Loaded preprocessed.csv ({len(texts)} rows)")
        return texts, labels
    try:
        raw = load_data(DATA_DIR)
    except TypeError:
        raw = load_data()
    if not raw:
        raise SystemExit(f"No reviews found under {DATA_DIR}.")
    print(f"Loaded {len(raw)} raw rows.")

    data = preprocess_reviews(raw)
    if not data:
        raise SystemExit("No labeled rows after preprocessing (neutral filtering?)")
    texts = [t for (t, _y) in data]
    labels = np.array([y for (_t, y) in data], dtype=int)
    print(f"Prepared {len(texts)} labeled rows.")
    return texts, labels


def main():
    texts, labels = _load_texts_labels()

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(
        texts, labels, VAL_SIZE, TEST_SIZE, SEED
    )
    print(
        f"Split sizes -> train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}"
    )

    # Train NB
    vec_nb = make_tfidf(
        ngram_max=NGRAM_MAX,
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=NB_MAX_FEATURES,
        sublinear_tf=True,
    )
    Xtr_nb, Xva_nb, Xte_nb = vectorize(vec_nb, X_train, X_val, X_test)
    print(
        f"NB vectorized shapes -> train: {Xtr_nb.shape} | val: {Xva_nb.shape} | test: {Xte_nb.shape}"
    )
    nb = MultinomialNB(alpha=NB_ALPHA)
    nb.fit(Xtr_nb, y_train)
    report(nb, "Validation (NB)", y_val, Xva_nb)
    report(nb, "Test (NB)", y_test, Xte_nb)
    save_artifacts("nb", vec_nb, nb)

    # Train LR
    vec_lr = make_tfidf(
        ngram_max=NGRAM_MAX,
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=LR_MAX_FEATURES,
        sublinear_tf=True,
    )
    Xtr_lr, Xva_lr, Xte_lr = vectorize(vec_lr, X_train, X_val, X_test)
    print(
        f"LR vectorized shapes -> train: {Xtr_lr.shape} | val: {Xva_lr.shape} | test: {Xte_lr.shape}"
    )
    logreg = LogisticRegression(
        C=LR_C,
        penalty=LR_PENALTY,
        solver=LR_SOLVER,
        class_weight=LR_CLASS_WEIGHT,
        max_iter=1000,
        n_jobs=None if LR_SOLVER == "liblinear" else -1,
        random_state=SEED,
    )
    logreg.fit(Xtr_lr, y_train)
    report(logreg, "Validation (LogReg)", y_val, Xva_lr)
    report(logreg, "Test (LogReg)", y_test, Xte_lr)
    save_artifacts("logreg", vec_lr, logreg)


if __name__ == "__main__":
    main()
