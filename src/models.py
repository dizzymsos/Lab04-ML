from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    implemented: bool
    pipeline: Pipeline | None
    student_note: str = ""


class RobustStackingClassifier(BaseEstimator, ClassifierMixin):
    """Stacking compacto para que el laboratorio sea robusto con clases pequeñas."""

    def __init__(
        self,
        random_state: int = 42,
        cv: int = 3,
        final_C: float = 1.0,
        tree_max_depth: int | None = 3,
        n_neighbors: int = 5,
        logistic_C: float = 1.0,
    ) -> None:
        self.random_state = random_state
        self.cv = cv
        self.final_C = final_C
        self.tree_max_depth = tree_max_depth
        self.n_neighbors = n_neighbors
        self.logistic_C = logistic_C

    def fit(self, X: Any, y: Any) -> "RobustStackingClassifier":
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        base_estimators = self._build_base_estimators()
        min_count = min(Counter(y_array).values())

        if min_count >= 2:
            meta_features = self._out_of_fold_meta_features(X, y_array, base_estimators, min_count)
        else:
            meta_features = self._in_sample_meta_features(X, y_array, base_estimators)

        self.final_estimator_ = LogisticRegression(
            C=self.final_C,
            class_weight="balanced",
            max_iter=5000,
            random_state=self.random_state,
        )
        self.final_estimator_.fit(meta_features, y_array)

        self.estimators_ = []
        for name, estimator in base_estimators:
            fitted = clone(estimator).fit(X, y_array)
            self.estimators_.append((name, fitted))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.final_estimator_.predict(self._meta_features_from_fitted(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.final_estimator_.predict_proba(self._meta_features_from_fitted(X))

    def _build_base_estimators(self) -> list[tuple[str, BaseEstimator]]:
        return [
            (
                "tree",
                DecisionTreeClassifier(
                    max_depth=self.tree_max_depth,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=self.random_state,
                ),
            ),
            (
                "knn",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            KNeighborsClassifier(
                                n_neighbors=self.n_neighbors,
                                weights="distance",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "logistic",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                C=self.logistic_C,
                                class_weight="balanced",
                                max_iter=5000,
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                ),
            ),
        ]

    def _out_of_fold_meta_features(
        self,
        X: Any,
        y_array: np.ndarray,
        base_estimators: list[tuple[str, BaseEstimator]],
        min_count: int,
    ) -> np.ndarray:
        n_splits = min(self.cv, min_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((len(y_array), len(base_estimators) * len(self.classes_)))

        for train_idx, valid_idx in cv.split(X, y_array):
            X_train = _safe_take(X, train_idx)
            X_valid = _safe_take(X, valid_idx)
            y_train = y_array[train_idx]

            for estimator_idx, (_, estimator) in enumerate(base_estimators):
                fitted = clone(estimator).fit(X_train, y_train)
                start = estimator_idx * len(self.classes_)
                end = start + len(self.classes_)
                meta_features[valid_idx, start:end] = self._aligned_predict_proba(fitted, X_valid)

        return meta_features

    def _in_sample_meta_features(
        self,
        X: Any,
        y_array: np.ndarray,
        base_estimators: list[tuple[str, BaseEstimator]],
    ) -> np.ndarray:
        fitted_estimators = [(name, clone(estimator).fit(X, y_array)) for name, estimator in base_estimators]
        return self._meta_features(X, fitted_estimators)

    def _meta_features_from_fitted(self, X: Any) -> np.ndarray:
        return self._meta_features(X, self.estimators_)

    def _meta_features(self, X: Any, estimators: list[tuple[str, BaseEstimator]]) -> np.ndarray:
        blocks = [self._aligned_predict_proba(estimator, X) for _, estimator in estimators]
        return np.hstack(blocks)

    def _aligned_predict_proba(self, estimator: BaseEstimator, X: Any) -> np.ndarray:
        proba = estimator.predict_proba(X)
        aligned = np.zeros((proba.shape[0], len(self.classes_)))
        estimator_classes = np.asarray(estimator.classes_)
        for source_idx, class_label in enumerate(estimator_classes):
            target_idx = int(np.where(self.classes_ == class_label)[0][0])
            aligned[:, target_idx] = proba[:, source_idx]
        return aligned


def _safe_take(X: Any, indices: np.ndarray) -> Any:
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return X[indices]


def build_model_registry(random_state: int = 42) -> dict[str, ModelSpec]:
    weak_tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
    )

    bagging = ModelSpec(
        key="bagging",
        display_name="Bagging",
        implemented=True,
        pipeline=Pipeline(
            [
                (
                    "clf",
                    BaggingClassifier(
                        estimator=DecisionTreeClassifier(
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                        n_estimators=50,
                        max_samples=0.8,
                        bootstrap=True,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    )

    adaboost = ModelSpec(
        key="adaboost",
        display_name="AdaBoost",
        implemented=True,
        pipeline=Pipeline(
            [
                (
                    "clf",
                    AdaBoostClassifier(
                        estimator=weak_tree,
                        n_estimators=50,
                        learning_rate=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    )

    stacking = ModelSpec(
        key="stacking",
        display_name="Stacking",
        implemented=True,
        pipeline=Pipeline(
            [
                (
                    "clf",
                    RobustStackingClassifier(
                        random_state=random_state,
                        cv=3,
                        final_C=1.0,
                        tree_max_depth=3,
                        n_neighbors=5,
                    ),
                ),
            ]
        ),
    )

    gradient_boosting = ModelSpec(
        key="gradient_boosting",
        display_name="Gradient Boosting",
        implemented=True,
        pipeline=Pipeline(
            [
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        subsample=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    )

    return {
        bagging.key: bagging,
        adaboost.key: adaboost,
        stacking.key: stacking,
        gradient_boosting.key: gradient_boosting,
    }
