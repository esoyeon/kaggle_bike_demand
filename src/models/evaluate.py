import yaml
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import os


def load_config():
    """설정 파일 로드"""
    with open("configs/model_config.yaml", "r") as f:
        return yaml.safe_load(f)


def setup_logging(config):
    """로깅 설정"""
    # 현재 시간으로 실행 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["logging"]["save_path"], f"evaluation_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 하위 디렉토리 생성
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)  # 그래프 저장용

    # 로그 파일 설정
    log_file = os.path.join(run_dir, "evaluation.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return run_dir


def calculate_metrics(y_true, y_pred):
    """
    다양한 평가 지표 계산

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        dict: 평가 지표들
    """
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred),
    }
    return metrics


def plot_residuals(y_true, y_pred, save_path):
    """잔차 분석 플롯"""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 잔차 vs 예측값
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color="r", linestyle="--")
    axes[0, 0].set_xlabel("Predicted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Predicted Values")

    # 잔차 히스토그램
    axes[0, 1].hist(residuals, bins=30, edgecolor="black")
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Residuals Distribution")

    # Q-Q plot
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")

    # 잔차의 절대값 vs 예측값
    axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.5)
    axes[1, 1].set_xlabel("Predicted Values")
    axes[1, 1].set_ylabel("Absolute Residuals")
    axes[1, 1].set_title("Scale Location Plot")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_prediction_analysis(y_true, y_pred, save_path):
    """예측 분석 플롯"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # 실제값 vs 예측값 산점도
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title("Actual vs Predicted Values")

    # 예측 오차의 분포
    error_pct = np.abs((y_true - y_pred) / y_true) * 100
    axes[1].hist(error_pct, bins=50, edgecolor="black")
    axes[1].set_xlabel("Percentage Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Prediction Error (%)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze_error_patterns(y_true, y_pred, features, save_path):
    """오차 패턴 분석"""
    error = np.abs(y_true - y_pred)
    error_df = pd.DataFrame({"error": error})

    # 특성별 오차 상관관계
    for col in features.columns:
        error_df[col] = features[col]

    correlations = error_df.corr()["error"].sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    correlations[1:11].plot(kind="bar")  # error 자신과의 상관관계 제외
    plt.title("Top 10 Feature Correlations with Prediction Error")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """메인 평가 프로세스"""
    config = load_config()
    run_dir = setup_logging(config)

    logging.info("Starting model evaluation...")

    try:
        # 데이터 로드 - config에서 경로 가져오기
        processed_data_path = config["data"]["processed_data_path"]
        df = pd.read_csv(processed_data_path)

        # 특성과 타겟 분리
        y_true = df["count"]
        X = df.drop(["count"], axis=1)
        logging.info(f"Data loaded. Shape: {X.shape}")

        # 가장 최근 실행 디렉토리 찾기
        logs_dir = Path(config["logging"]["save_path"])
        run_folders = [f for f in logs_dir.glob("run_*") if f.is_dir()]
        if not run_folders:
            raise FileNotFoundError("No run directories found")

        latest_run = max(run_folders, key=lambda x: x.stat().st_mtime)
        model_dir = latest_run / "models"

        # 해당 디렉토리에서 모델 파일 찾기
        model_files = list(model_dir.glob("*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")

        # 가장 좋은 성능의 모델 찾기 (파일 이름에 포함된 성능 지표 사용)
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logging.info(f"Loading model from: {latest_model}")
        model = joblib.load(latest_model)

        # 예측 수행
        y_pred = model.predict(X)

        # 메트릭 계산
        metrics = calculate_metrics(y_true, y_pred)

        # 결과 로깅
        logging.info("\n=== Model Performance Metrics ===")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")

        # 시각화
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 잔차 분석
        residuals_path = os.path.join(run_dir, "plots", f"residuals_{timestamp}.png")
        plot_residuals(y_true, y_pred, residuals_path)
        logging.info(f"Residual analysis plots saved to {residuals_path}")

        # 예측 분석
        predictions_path = os.path.join(
            run_dir, "plots", f"predictions_{timestamp}.png"
        )
        plot_prediction_analysis(y_true, y_pred, predictions_path)
        logging.info(f"Prediction analysis plots saved to {predictions_path}")

        # 오차 패턴 분석
        error_patterns_path = os.path.join(
            run_dir, "plots", f"error_patterns_{timestamp}.png"
        )
        analyze_error_patterns(y_true, y_pred, X, error_patterns_path)
        logging.info(f"Error pattern analysis saved to {error_patterns_path}")

    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise

    logging.info("Model evaluation completed!")


if __name__ == "__main__":
    main()
