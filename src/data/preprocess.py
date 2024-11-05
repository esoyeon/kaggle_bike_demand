import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging


def load_raw_data(file_path):
    """원본 데이터 로드"""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """데이터 전처리"""
    # datetime 처리
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek

    # 범주형 변수 원핫 인코딩
    season_dummies = pd.get_dummies(df["season"], prefix="season")
    weather_dummies = pd.get_dummies(df["weather"], prefix="weather")

    # 데이터프레임 결합
    df = pd.concat([df, season_dummies, weather_dummies], axis=1)

    # 불필요한 컬럼 제거 (casual과 registered는 count의 하위 항목이므로 제거)
    columns_to_drop = ["datetime", "season", "weather", "casual", "registered"]
    df = df.drop(columns_to_drop, axis=1)

    # 타겟 변수 분리
    if "count" in df.columns:
        y = df.pop("count")  # count 컬럼을 추출하고 제거
        return df, y
    else:
        return df, None


def save_processed_data(df, y, output_path):
    """전처리된 데이터 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 데이터 확인
    logging.info(f"Processed features shape: {df.shape}")
    logging.info(f"Target shape: {y.shape}")

    # 저장 전 데이터 결합
    final_df = pd.concat([df, pd.Series(y, name="count")], axis=1)
    logging.info(f"Final dataframe shape: {final_df.shape}")
    logging.info("Sample of processed data:")
    logging.info(final_df.head())

    final_df.to_csv(output_path, index=False)
    logging.info(f"Saved processed data to {output_path}")


def main():
    """메인 전처리 프로세스"""
    # 데이터 로드
    input_path = "data/raw/bike_train.csv"
    output_path = "data/processed/bike_processed.csv"

    df = pd.read_csv(input_path)
    logging.info(f"Original data shape: {df.shape}")
    logging.info("Original columns:")
    logging.info(df.columns.tolist())

    # 전처리
    X, y = preprocess_data(df)

    # 전처리된 데이터 저장
    save_processed_data(X, y, output_path)


if __name__ == "__main__":
    main()
