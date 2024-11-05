import yaml
import logging
import os
from datetime import datetime
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_config():
    """설정 파일 로드"""
    with open("configs/model_config.yaml", "r") as f:
        return yaml.safe_load(f)


def setup_logging(config):
    """로깅 설정"""
    # 현재 시간으로 실행 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["logging"]["save_path"], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 하위 디렉토리 생성
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)  # 모델 저장용
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)  # 그래프 저장용

    # 로그 파일 설정
    log_file = os.path.join(run_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return run_dir  # 생성된 실행 디렉토리 경로 반환


def load_data(config):
    """처리된 특성 데이터 로드"""
    features_path = config["data"]["processed_data_path"]
    df = pd.read_csv(features_path)

    logging.info("Data columns:")
    logging.info(df.columns.tolist())

    # 타겟 분리 전 데이터 확인
    logging.info(f"Data shape before split: {df.shape}")
    logging.info(
        f"Sample of target values: {df['count'].head() if 'count' in df.columns else 'No count column found'}"
    )

    # 타겟 분리
    if "count" not in df.columns:
        raise ValueError("Target column 'count' not found in the data")

    X = df.drop(["count"], axis=1)
    y = df["count"]

    # 분리 후 데이터 확인
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Target shape: {y.shape}")

    return X, y


def create_model(model_config, random_state=42):
    """모델 객체 생성"""
    if model_config["name"] == "Linear Regression":
        return LinearRegression()
    elif model_config["name"] == "Ridge":
        return Ridge(random_state=random_state)
    elif model_config["name"] == "Lasso":
        return Lasso(random_state=random_state)
    elif model_config["name"] == "Random Forest":
        return RandomForestRegressor(random_state=random_state)
    elif model_config["name"] == "XGBoost":
        return xgb.XGBRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")


def train_model(model, X_train, y_train, model_config, config):
    """모델 학습 및 하이퍼파라미터 튜닝"""
    if "params" not in model_config:
        model.fit(X_train, y_train)
        return model

    # 파라미터 수와 반복 횟수 축소
    n_iter = 5  # 20에서 5로 감소
    cv = 3  # 5에서 3으로 감소

    # 파라미터 범위 축소
    if model_config["name"] == "Random Forest":
        param_distributions = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5],
        }
    elif model_config["name"] == "XGBoost":
        param_distributions = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.1],
            "subsample": [0.8],
        }
    elif model_config["name"] in ["Ridge", "Lasso"]:
        param_distributions = {"alpha": [0.1, 1.0]}
    else:
        model.fit(X_train, y_train)
        return model

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,  # 병렬 처리 활성화
    )

    random_search.fit(X_train, y_train)
    logging.info(f"Best parameters: {random_search.best_params_}")

    return random_search.best_estimator_


def evaluate_model(model, X, y, dataset_name=""):
    """모델 성능 평가"""
    y_pred = model.predict(X)

    # 예측값 범위 확인
    logging.info(f"\nPrediction statistics for {dataset_name}:")
    logging.info(
        f"Actual values - min: {y.min():.2f}, max: {y.max():.2f}, mean: {y.mean():.2f}"
    )
    logging.info(
        f"Predicted values - min: {y_pred.min():.2f}, max: {y_pred.max():.2f}, mean: {y_pred.mean():.2f}"
    )

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    logging.info(f"\n{dataset_name} Set Performance:")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"R2 Score: {r2:.4f}")

    # 잔차 분석
    residuals = y - y_pred
    logging.info(
        f"Residuals - mean: {residuals.mean():.4f}, std: {residuals.std():.4f}"
    )

    return {"rmse": rmse, "r2": r2, "predictions": y_pred}


def plot_predictions(y_true, y_pred, title, save_path):
    """예측 결과 시각화"""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_true)), y_true, "b-", label="Actual", alpha=0.5)
    plt.plot(range(len(y_pred)), y_pred, "r-", label="Predicted", alpha=0.5)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Bike Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_model(model, model_name, config):
    """모델 저장"""
    save_dir = config["logging"]["model_save_path"]
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"{model_name}_{timestamp}.joblib")

    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")


def main():
    """메인 학습 프로세스"""
    # 설정 로드
    config = load_config()

    # 로깅 설정 및 실행 디렉토리 얻기
    run_dir = setup_logging(config)
    logging.info(f"Created run directory: {run_dir}")

    logging.info("Starting model training process...")

    # 데이터 로드
    X, y = load_data(config)
    logging.info(f"Data loaded. Shape: {X.shape}")

    # 모든 모델 학습
    for model_type in ["linear", "tree"]:
        for model_config in config["models"][model_type]:
            logging.info(f"\nTraining {model_config['name']}...")

            # 모델 생성
            model = create_model(model_config)

            # 모델 학습
            trained_model = train_model(model, X, y, model_config, config)

            # 모델 평가
            results = evaluate_model(trained_model, X, y, "Full")

            # 예측 결과 시각화
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(run_dir, "plots", f"predictions_{timestamp}.png")
            plot_predictions(
                y,
                results["predictions"],
                f"{model_config['name']} Predictions",
                plot_path,
            )

            # 모델 저장
            model_path = os.path.join(
                run_dir,
                "models",
                f'{model_config["name"].lower().replace(" ", "_")}_{timestamp}.joblib',
            )
            joblib.dump(trained_model, model_path)
            logging.info(f"Model saved to {model_path}")

    logging.info("\nTraining process completed!")


if __name__ == "__main__":
    main()
