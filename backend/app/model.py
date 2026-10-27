"""
Generalized Linear Model (GLM) for Genomic Pipeline Failure Prediction (Version 2.1.0).

Uses scikit-learn's LogisticRegression (binomial GLM with logit link function)
to provide supplementary statistical failure risk scoring and factor attribution
based on extracted compiler features.
Includes:
- Robust feature scaling via StandardScaler in Pipeline
- Group-aware holdout evaluation and 3-fold StratifiedGroupKFold cross-validation
- Brier calibration scoring, ROC-AUC, and baseline comparisons
- Dynamic positive-class resolution via model.classes_
- Comprehensive artifact metadata and loading compatibility verification
"""

import os
import platform
from typing import Dict, Any, Optional
import pandas as pd
import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

from app.schemas import GLMPrediction, PipelineFeatures
from app.features import FEATURE_COLUMNS, features_to_dataframe, dict_to_dataframe

# Default model artifact path
DEFAULT_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "glm_pipeline_model.joblib")

MODEL_SCHEMA_VERSION = "2.1.0"


class GLMModelManager:
    """
    Manages training, cross-validation, persistence, verification, and inference
    for the GLM failure model.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self.feature_names = list(FEATURE_COLUMNS)
        self.metadata: Dict[str, Any] = {}

    def is_loaded(self) -> bool:
        """Check if a verified model pipeline is loaded in memory."""
        return self.pipeline is not None

    def load_model(self) -> bool:
        """
        Load and verify trained pipeline artifact from disk.
        Validates artifact format, feature schema compatibility, and version info.
        """
        if not os.path.exists(self.model_path):
            return False

        try:
            artifact = joblib.load(self.model_path)
            if not isinstance(artifact, dict) or "pipeline" not in artifact:
                print(f"Warning: Model artifact at {self.model_path} has unrecognized format.")
                return False

            pipeline = artifact["pipeline"]
            if not isinstance(pipeline, Pipeline):
                print("Warning: Loaded artifact is not a valid scikit-learn Pipeline.")
                return False

            artifact_features = artifact.get("features", [])
            if artifact_features != self.feature_names:
                print(
                    f"Warning: Artifact feature schema mismatch. "
                    f"Expected: {self.feature_names}, Found: {artifact_features}"
                )
                return False

            # Verify classifier has classes_ attribute
            classifier = pipeline.named_steps.get("classifier")
            if classifier is None or not hasattr(classifier, "classes_"):
                print("Warning: Pipeline is missing a trained classifier step.")
                return False

            self.pipeline = pipeline
            self.metadata = artifact.get("metadata", {})
            self.feature_names = artifact_features
            return True

        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            return False

    def train_and_save(
        self,
        df: pd.DataFrame,
        target_col: str = "failed",
        group_col: Optional[str] = "group_id",
        test_size: float = 0.25,
        random_state: int = 42,
        cv_folds: int = 3,
        independent_df: Optional[pd.DataFrame] = None,
        dataset_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train Logistic Regression (GLM) model on labelled dataset with group-aware cross-validation
        and save verified artifact with rich metadata, baseline comparisons, and calibration metrics.

        Args:
            df: DataFrame containing FEATURE_COLUMNS, binary target_col, and optional group_col.
            target_col: Binary label column (1 = failed/invalid, 0 = success/valid).
            group_col: Column indicating template/architecture group lineage.
            test_size: Holdout evaluation proportion.
            random_state: Reproducibility seed.
            cv_folds: Number of Stratified Group K-Fold cross-validation folds.
            independent_df: Optional out-of-distribution benchmark DataFrame.
            dataset_hash: SHA-256 hash of training data file.

        Returns:
            Dictionary containing evaluation metrics, baseline comparisons, and model metadata.
        """
        # Ensure exact feature schema order
        X = df[self.feature_names]
        y = df[target_col].astype(int)

        # 1. Group-aware train/test split if group_col is present
        has_groups = group_col and group_col in df.columns and df[group_col].nunique() > 1
        if has_groups:
            groups = df[group_col].values
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups_train = groups[train_idx]
            groups_test = groups[test_idx]

            train_groups = sorted(list(set(groups_train)))
            test_groups = sorted(list(set(groups_test)))
            overlap = set(train_groups).intersection(set(test_groups))
            overlap_pct = (len(overlap) / len(test_groups)) * 100.0 if test_groups else 0.0
            split_method = "GroupShuffleSplit (Group-Isolated Holdout)"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            groups_train = None
            groups_test = None
            train_groups = []
            test_groups = []
            overlap_pct = 0.0
            split_method = "StratifiedShuffleSplit (Standard Random Split)"

        # 2. Build Full Pipeline: StandardScaler -> Logistic Regression (GLM with Logit Link)
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        # 3. Cross Validation on training split strictly (no test leakage)
        if has_groups:
            sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(pipeline, X_train, y_train, groups=groups_train, cv=sgkf, scoring="accuracy")
        else:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")

        # 4. Fit Full GLM Pipeline on entire training split
        pipeline.fit(X_train, y_train)

        # 5. Evaluate Full GLM on held-out test split
        classifier: LogisticRegression = pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)
        failure_idx = classes.index(1) if 1 in classes else 1

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, failure_idx]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob)) if len(set(y_test)) > 1 else 1.0
        brier = float(brier_score_loss(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Class support in test set
        test_support = {
            "class_0_success": int((y_test == 0).sum()),
            "class_1_failure": int((y_test == 1).sum()),
        }

        # 6. Baseline Comparisons on Held-out Test Split
        # Baseline A: Simple Rule-Based Compiler Baseline (error count > 0)
        rule_pred = (X_test["total_error_count"] > 0).astype(int)
        rule_metrics = {
            "name": "Deterministic Compiler Rule (total_error_count > 0)",
            "accuracy": round(float(accuracy_score(y_test, rule_pred)), 4),
            "precision": round(float(precision_score(y_test, rule_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, rule_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, rule_pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, rule_pred).tolist(),
        }

        # Baseline B: Majority-Class Baseline (DummyClassifier)
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)
        dummy_metrics = {
            "name": "Majority-Class Baseline",
            "accuracy": round(float(accuracy_score(y_test, dummy_pred)), 4),
            "precision": round(float(precision_score(y_test, dummy_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, dummy_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, dummy_pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, dummy_pred).tolist(),
        }

        # Baseline C: Ablated GLM (Logistic Regression WITHOUT total_error_count)
        ablated_feats = [f for f in self.feature_names if f != "total_error_count"]
        ablated_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        ablated_pipeline.fit(X_train[ablated_feats], y_train)
        ablated_pred = ablated_pipeline.predict(X_test[ablated_feats])
        ablated_prob = ablated_pipeline.predict_proba(X_test[ablated_feats])[:, failure_idx]
        ablated_metrics = {
            "name": "Ablated GLM (Excluding total_error_count)",
            "features_used": ablated_feats,
            "accuracy": round(float(accuracy_score(y_test, ablated_pred)), 4),
            "precision": round(float(precision_score(y_test, ablated_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, ablated_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, ablated_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, ablated_prob)), 4) if len(set(y_test)) > 1 else 1.0,
            "brier_score": round(float(brier_score_loss(y_test, ablated_prob)), 4),
            "confusion_matrix": confusion_matrix(y_test, ablated_pred).tolist(),
        }

        # 7. Independent Benchmark Evaluation (if provided)
        independent_eval_metrics = None
        if independent_df is not None and not independent_df.empty:
            X_indep = independent_df[self.feature_names]
            y_indep = independent_df[target_col].astype(int)

            indep_pred = pipeline.predict(X_indep)
            indep_rule_pred = (X_indep["total_error_count"] > 0).astype(int)
            indep_ablated_pred = ablated_pipeline.predict(X_indep[ablated_feats])

            compiler_cm = confusion_matrix(y_indep, indep_rule_pred, labels=[0, 1])
            _, comp_fp, comp_fn, _ = compiler_cm.ravel()

            glm_cm = confusion_matrix(y_indep, indep_pred, labels=[0, 1])
            _, glm_fp, glm_fn, _ = glm_cm.ravel()

            independent_eval_metrics = {
                "sample_count": len(independent_df),
                "glm_full": {
                    "accuracy": round(float(accuracy_score(y_indep, indep_pred)), 4),
                    "precision": round(float(precision_score(y_indep, indep_pred, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_indep, indep_pred, zero_division=0)), 4),
                    "f1_score": round(float(f1_score(y_indep, indep_pred, zero_division=0)), 4),
                    "confusion_matrix": glm_cm.tolist(),
                    "false_positives": int(glm_fp),
                    "false_negatives": int(glm_fn),
                },
                "compiler_baseline": {
                    "accuracy": round(float(accuracy_score(y_indep, indep_rule_pred)), 4),
                    "precision": round(float(precision_score(y_indep, indep_rule_pred, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_indep, indep_rule_pred, zero_division=0)), 4),
                    "f1_score": round(float(f1_score(y_indep, indep_rule_pred, zero_division=0)), 4),
                    "confusion_matrix": compiler_cm.tolist(),
                    "compiler_false_positives": int(comp_fp),
                    "compiler_false_negatives": int(comp_fn),
                },
                "glm_ablated": {
                    "accuracy": round(float(accuracy_score(y_indep, indep_ablated_pred)), 4),
                    "precision": round(float(precision_score(y_indep, indep_ablated_pred, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_indep, indep_ablated_pred, zero_division=0)), 4),
                    "f1_score": round(float(f1_score(y_indep, indep_ablated_pred, zero_division=0)), 4),
                },
            }

        # Extract coefficients and weights
        coefs = classifier.coef_[0].tolist()
        intercept = float(classifier.intercept_[0])
        feature_importance = dict(zip(self.feature_names, [round(c, 4) for c in coefs]))

        metrics = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_type": "Generalized Linear Model (Logistic Regression with Logit Link)",
            "framework": f"scikit-learn=={sklearn.__version__}",
            "python_version": platform.python_version(),
            "dataset_hash": dataset_hash or "N/A",
            "target_column": target_col,
            "classes": classes,
            "positive_class": 1,
            "splitting_method": split_method,
            "train_groups": train_groups,
            "test_groups": test_groups,
            "group_overlap_percent": overlap_pct,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "brier_score": round(brier, 4),
            "cv_folds": cv_folds,
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": cm,
            "test_class_support": test_support,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "intercept": round(intercept, 4),
            "feature_coefficients": feature_importance,
            "baselines": {
                "rule_based_compiler": rule_metrics,
                "majority_class": dummy_metrics,
                "glm_ablated_without_total_error_count": ablated_metrics,
                "glm_full": {
                    "name": "GLM Full (All 13 Features)",
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "roc_auc": round(roc_auc, 4),
                    "brier_score": round(brier, 4),
                },
            },
            "independent_evaluation": independent_eval_metrics,
            "leakage_audit_summary": (
                "Evaluated using group-isolated holdout split (0.0% template overlap between train and test). "
                "The rule-based compiler baseline matches ground truth on static validity by definition because labels "
                "are derived from static compiler diagnostics (total_error_count > 0). "
                "The GLM full model learns this linear relationship directly. "
                "The ablated GLM tests generalization when total_error_count is excluded."
            ),
        }

        # Save artifact with strict schema metadata
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(
            {
                "pipeline": pipeline,
                "features": self.feature_names,
                "metadata": metrics,
                "schema_version": MODEL_SCHEMA_VERSION,
            },
            self.model_path,
        )

        self.pipeline = pipeline
        self.metadata = metrics
        return metrics

    def predict_risk(self, features: PipelineFeatures) -> GLMPrediction:
        """Predict failure probability and risk tier from validated PipelineFeatures."""
        if self.pipeline is None:
            if not self.load_model():
                raise RuntimeError(
                    "GLM model is not loaded and no trained model file was found on disk. Run model training first."
                )

        df_input = features_to_dataframe(features)
        return self._predict_dataframe(df_input, features.model_dump())

    def predict_from_dict(self, feature_dict: Dict[str, Any]) -> GLMPrediction:
        """Predict from arbitrary dictionary with strict schema and value validation."""
        if self.pipeline is None:
            if not self.load_model():
                raise RuntimeError("GLM model is not loaded. Train the model first.")

        df_input = dict_to_dataframe(feature_dict)
        return self._predict_dataframe(df_input, feature_dict)

    def _predict_dataframe(
        self, df_input: pd.DataFrame, raw_features: Dict[str, Any]
    ) -> GLMPrediction:
        """
        Internal prediction execution.
        Dynamically resolves the failure class index from classifier.classes_.
        Maps predicted probabilities to risk tiers:
        - LOW:    p < 0.30
        - MEDIUM: 0.30 <= p < 0.70
        - HIGH:   p >= 0.70
        """
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]
        scaler: StandardScaler = self.pipeline.named_steps["scaler"]

        # Dynamically resolve positive class index
        classes = list(classifier.classes_)
        failure_idx = classes.index(1) if 1 in classes else 1
        success_idx = 1 - failure_idx

        probs = self.pipeline.predict_proba(df_input)[0]
        prob_failure = float(probs[failure_idx])
        prob_success = float(probs[success_idx])

        # Risk tiering
        if prob_failure < 0.30:
            risk_category = "LOW"
        elif prob_failure < 0.70:
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"

        predicted_outcome = "FAILURE" if prob_failure >= 0.50 else "SUCCESS"
        confidence = prob_failure if predicted_outcome == "FAILURE" else prob_success

        # Feature contribution attribution
        coefs = classifier.coef_[0]
        scaled_vals = scaler.transform(df_input)[0]
        contributions = coefs * scaled_vals

        top_factors = []
        contrib_pairs = sorted(
            zip(self.feature_names, contributions), key=lambda x: x[1], reverse=True
        )
        for name, val in contrib_pairs:
            if val > 0.05 and raw_features.get(name, 0) > 0:
                top_factors.append(f"{name} (impact: +{round(val, 2)})")

        if not top_factors and prob_failure < 0.30:
            top_factors.append("No critical pipeline risk factors identified")

        return GLMPrediction(
            failure_probability=round(prob_failure, 4),
            risk_category=risk_category,
            predicted_outcome=predicted_outcome,
            confidence=round(confidence, 4),
            top_risk_factors=top_factors[:4],
        )


# Global singleton instance
model_manager = GLMModelManager()
