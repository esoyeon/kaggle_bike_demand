import yaml
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import os


def load_config():
    """설정 파일 로드"""
    with open("configs/model_config.yaml", "r") as f:
        return yaml.safe_load(f)


def setup_logging(config):
    """로깅 설정"""
    # 현재 시간으로 실행 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["logging"]["save_path"], f"prediction_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 하위 디렉토리 생성
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)  # 예측 결과 저장용
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)  # 그래프 저장용

    # 로그 파일 설정
    log_file = os.path.join(run_dir, "prediction.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return run_dir


def load_model_and_transformers(config, model_name):
    """
    저장된 모델과 전처리 변환기 로드

    Args:
        config: 설정 정보
        model_name: 모델 파일 이름

    Returns:
        tuple: (모델, 스케일러, 다항식 변환기)
    """
    model_path = Path(config["logging"]["model_save_path"]) / model_name
    feature_path = Path("data/processed/features")

    model = joblib.load(model_path)
    scaler = joblib.load(feature_path / "scaler.joblib")
    poly = joblib.load(feature_path / "polynomial.joblib")

    return model, scaler, poly


def prepare_features(data, scaler, poly=None, is_linear=False):
    """
    예측을 위한 특성 준비

    Args:
        data: 입력 데이터
        scaler: 스케일러
        poly: 다항식 변환기 (선형 모델용)
        is_linear: 선형 모델 여부

    Returns:
        array: 처리된 특성
    """
    # 스케일링
    data_scaled = scaler.transform(data)

    # 선형 모델이고 다항식 변환기가 있는 경우
    if is_linear and poly is not None:
        return poly.transform(data_scaled)

    return data_scaled


def make_predictions(model, X):
    """
    예측 수행

    Args:
        model: 학습된 모델
        X: 입력 특성

    Returns:
        array: 예측값
    """
    return model.predict(X)


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


def main():
    """메인 예측 프로세스"""
    # 설정 로드
    config = load_config()
    run_dir = setup_logging(config)

    logging.info("Starting prediction process...")

    try:
        # 데이터 로드 - config에서 경로 가져오기
        processed_data_path = config["data"]["processed_data_path"]
        df = pd.read_csv(processed_data_path)

        # 특성과 타겟 분리 (있는 경우)
        if "count" in df.columns:
            y_true = df["count"]
            X = df.drop(["count"], axis=1)
        else:
            X = df
            y_true = None

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

        # 가장 최근 모델 로드
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logging.info(f"Loading model from: {latest_model}")
        model = joblib.load(latest_model)

        # 예측 수행
        predictions = model.predict(X)
        logging.info("Predictions generated successfully")

        # 결과 저장
        results_df = pd.DataFrame({"predicted_count": predictions})

        # 예측 결과 저장 디렉토리 생성
        predictions_dir = os.path.join(run_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(predictions_dir, f"predictions_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        logging.info(f"Predictions saved to {results_path}")

        # 예측 결과 시각화 (실제 값이 있는 경우)
        if y_true is not None:
            plot_path = os.path.join(run_dir, "plots", f"predictions_{timestamp}.png")
            plot_predictions(y_true, predictions, "Prediction Results", plot_path)
            logging.info(f"Prediction plot saved to {plot_path}")

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise

    logging.info("Prediction process completed!")


if __name__ == "__main__":
    main()
