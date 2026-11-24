import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42


# ==============================================================================
# 1. HÀM ĐÁNH GIÁ MÔ HÌNH QUA STRATIFIED 10-FOLD CV (KHÔNG DATA LEAKAGE)
# ==============================================================================
def evaluate_model_cv(model, X, y, use_smote=True, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = {
        "Recall": [],
        "Precision": [],
        "F1": [],
        "ROC-AUC": [],
        "PR-AUC": [],
        "Accuracy": [],
    }

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Đóng gói Scaler và SMOTE vào Pipeline để chỉ fit trên tập Train
        steps = [("scaler", StandardScaler())]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("clf", model))

        pipeline = ImbPipeline(steps=steps)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        metrics["Recall"].append(recall_score(y_val, y_pred, zero_division=0))
        metrics["Precision"].append(precision_score(y_val, y_pred, zero_division=0))
        metrics["F1"].append(f1_score(y_val, y_pred, zero_division=0))
        metrics["ROC-AUC"].append(roc_auc_score(y_val, y_prob))
        metrics["PR-AUC"].append(average_precision_score(y_val, y_prob))
        metrics["Accuracy"].append(accuracy_score(y_val, y_pred))

    summary = {
        k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()
    }
    raw_means = {k: np.mean(v) for k, v in metrics.items()}
    return summary, raw_means


# ==============================================================================
# 2. TINH CHỈNH SIÊU THAM SỐ BẰNG OPTUNA (TỐI ƯU HÓA TỔNG HỢP RECALL & ROC-AUC)
# ==============================================================================
def tune_xgboost(X, y, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        _, means = evaluate_model_cv(model, X, y, use_smote=True, n_splits=10)
        # Ưu tiên kết hợp Recall (0.6) và ROC-AUC (0.4) cho bài toán y tế
        return 0.6 * means["Recall"] + 0.4 * means["ROC-AUC"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss"})
    return XGBClassifier(**best_params), best_params


def tune_lightgbm(X, y, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 25),
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        _, means = evaluate_model_cv(model, X, y, use_smote=True, n_splits=10)
        return 0.6 * means["Recall"] + 0.4 * means["ROC-AUC"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params.update({"random_state": RANDOM_STATE, "verbose": -1})
    return LGBMClassifier(**best_params), best_params


# ==============================================================================
# 3. CHẠY THỰC NGHIỆM SO SÁNH & KẾT HỢP MÔ HÌNH (SOFT VOTING)
# ==============================================================================
def run_experiment(df, scenario_name, drop_time=False):
    print(f"\n{'='*80}\nKỊCH BẢN: {scenario_name}\n{'='*80}")
    data = df.copy()
    if drop_time and "time" in data.columns:
        data = data.drop(columns=["time"])

    X = data.drop(columns=["DEATH_EVENT"])
    y = data["DEATH_EVENT"]

    # Tỷ lệ mất cân bằng lớp để cấu hình scale_pos_weight khi không dùng SMOTE
    scale_weight = (y == 0).sum() / (y == 1).sum()

    # Bước A: Huấn luyện các mô hình Baseline & Mặc định
    models = {
        "1. Logistic Regression (Baseline)": (LogisticRegression(random_state=RANDOM_STATE), True),
        "2. Random Forest (Baseline)": (RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), True),
        "3. XGBoost (Class Weight - No SMOTE)": (
            XGBClassifier(scale_pos_weight=scale_weight, random_state=RANDOM_STATE, eval_metric="logloss"),
            False,
        ),
        "4. XGBoost (Default + SMOTE)": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            True,
        ),
        "5. LightGBM (Default + SMOTE)": (
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            True,
        ),
        "6. CatBoost (Default + SMOTE)": (
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
            True,
        ),
    }

    results = []
    for name, (model, use_smote) in models.items():
        summary, _ = evaluate_model_cv(model, X, y, use_smote=use_smote)
        summary["Mô hình"] = name
        results.append(summary)

    # Bước B: Cải tiến với Optuna Hyperparameter Tuning
    print("-> Đang chạy Optuna tinh chỉnh XGBoost & LightGBM...")
    tuned_xgb, xgb_params = tune_xgboost(X, y, n_trials=30)
    tuned_lgbm, lgbm_params = tune_lightgbm(X, y, n_trials=30)

    for name, model in [
        ("7. XGBoost (Optuna Tuned + SMOTE)", tuned_xgb),
        ("8. LightGBM (Optuna Tuned + SMOTE)", tuned_lgbm),
    ]:
        summary, _ = evaluate_model_cv(model, X, y, use_smote=True)
        summary["Mô hình"] = name
        results.append(summary)

    # Bước C: Cải tiến kết hợp mô hình (Soft Voting Ensemble)
    voting_clf = VotingClassifier(
        estimators=[
            ("xgb", tuned_xgb),
            ("lgbm", tuned_lgbm),
            ("cat", CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)),
        ],
        voting="soft",
    )
    summary_voting, _ = evaluate_model_cv(voting_clf, X, y, use_smote=True)
    summary_voting["Mô hình"] = "9. Soft Voting Ensemble (XGB+LGBM+Cat)"
    results.append(summary_voting)

    # Hiển thị bảng kết quả
    cols = ["Mô hình", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC", "Accuracy"]
    res_df = pd.DataFrame(results)[cols]
    print(res_df.to_string(index=False))

    # Đóng gói và lưu mô hình tốt nhất trên toàn bộ tập dữ liệu để dùng cho Streamlit & SHAP
    final_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", voting_clf),
    ])
    final_pipeline.fit(X, y)
    tag = "no_time" if drop_time else "full_features"
    joblib.dump(final_pipeline, f"heart_failure_model_{tag}.joblib")
    print(f"-> Đã lưu mô hình hoàn chỉnh tại: heart_failure_model_{tag}.joblib")


if __name__ == "__main__":
    # Thay đường dẫn bằng file csv 
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")
    # Kịch bản 1: Đầy đủ 12 đặc trưng (Đối chiếu với bài báo gốc Chicco & Jurman 2020)
    run_experiment(df, "ĐẦY ĐỦ 12 ĐẶC TRƯNG (BAO GỒM BIẾN TIME", drop_time=False)

    # Kịch bản 2: Loại bỏ biến time (11 chỉ số lâm sàng tại thời điểm nhập viện)
    run_experiment(df, "11 ĐẶC TRƯNG LÂM SÀNG (LOẠI BỎ BIẾN TIME)", drop_time=True)