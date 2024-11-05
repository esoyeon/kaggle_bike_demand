import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib
import os
from pathlib import Path


def create_time_features(df):
    """
    시간 관련 특성 생성

    Args:
        df: 입력 데이터프레임

    Returns:
        pd.DataFrame: 시간 특성이 추가된 데이터프레임
    """
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def create_weather_features(df):
    """
    날씨 관련 특성 생성

    Args:
        df: 입력 데이터프레임

    Returns:
        pd.DataFrame: 날씨 특성이 추가된 데이터프레임
    """
    # 온도와 체감온도의 차이
    df["temp_diff"] = df["temp"] - df["atemp"]

    # 습도와 온도의 상호작용
    df["humid_temp"] = df["humidity"] * df["temp"]

    return df


def scale_features(df, scaler=None, is_training=True):
    """
    특성 스케일링

    Args:
        df: 입력 데이터프레임
        scaler: 기존 스케일러 (없으면 새로 생성)
        is_training: 학습용 데이터 여부

    Returns:
        tuple: (스케일링된 데이터프레임, 스케일러)
    """
    features_to_scale = [
        "temp",
        "atemp",
        "humidity",
        "windspeed",
        "temp_diff",
        "humid_temp",
    ]

    if is_training:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features_to_scale])
    else:
        scaled_features = scaler.transform(df[features_to_scale])

    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaled_features

    return df_scaled, scaler


def create_polynomial_features(df, degree=2):
    """
    다항식 특성 생성 (선형 모델용)

    Args:
        df: 입력 데이터프레임
        degree: 다항식 차수

    Returns:
        tuple: (다항식 특성이 추가된 데이터프레임, 변환기)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df)
    feature_names = poly.get_feature_names_out(df.columns)

    return pd.DataFrame(poly_features, columns=feature_names), poly


def save_features(df, scaler, poly, output_dir):
    """
    특성 데이터와 변환기 저장

    Args:
        df: 처리된 데이터프레임
        scaler: 학습된 스케일러
        poly: 학습된 다항식 변환기
        output_dir: 저장 경로
    """
    # 저장 경로 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 데이터 저장
    df.to_csv(output_path / "processed_features.csv", index=False)

    # 변환기 저장
    joblib.dump(scaler, output_path / "scaler.joblib")
    joblib.dump(poly, output_path / "polynomial.joblib")


def main():
    """
    특성 엔지니어링 메인 함수
    """
    # 전처리된 데이터 로드
    input_path = "data/processed/bike_processed.csv"
    output_dir = "data/processed/features"
    df = pd.read_csv(input_path)

    # 특성 생성
    df = create_time_features(df)
    df = create_weather_features(df)

    # 스케일링
    df_scaled, scaler = scale_features(df, is_training=True)

    # 다항식 특성 생성
    df_poly, poly = create_polynomial_features(df_scaled)

    # 결과 저장
    save_features(df_poly, scaler, poly, output_dir)


if __name__ == "__main__":
    main()
