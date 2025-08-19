from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def stratified_splits(
    texts: Sequence[str],
    labels: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
):
    """
    Create stratified train/val/test splits.
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    val_frac = val_size / (1.0 - test_size) if test_size < 1.0 else 0.0
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_tfidf(
    ngram_max: int,
    min_df,
    max_df,
    max_features: Optional[int] = None,
    sublinear_tf: bool = True,
):
    """
    Construct a TfidfVectorizer with common defaults used in trainers.
    """
    return TfidfVectorizer(
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )


def vectorize(vec: TfidfVectorizer, X_train, X_val, X_test):
    """
    Fit the vectorizer on X_train and transform all splits.
    Returns (Xtr, Xva, Xte).
    """
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)
    Xte = vec.transform(X_test)
    return Xtr, Xva, Xte


def report(model, split_name: str, y_true, Xmat) -> None:
    """
    Print accuracy and a classification report for a given split.
    """
    y_pred = model.predict(Xmat)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{split_name} accuracy: {acc:.4f}")
    print(f"{split_name} report:\n{classification_report(y_true, y_pred, digits=4)}")


def save_artifacts(prefix: str, vectorizer, model) -> None:
    """
    Save vectorizer and model.
    """
    Path("models").mkdir(exist_ok=True)
    vec_path = f"models/tfidf_{prefix}.joblib"
    model_path = f"models/{prefix}_model.joblib"
    dump(vectorizer, vec_path)
    dump(model, model_path)
    print(f"\nSaved: {vec_path} and {model_path}")
