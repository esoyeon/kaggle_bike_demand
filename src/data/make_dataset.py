import pandas as pd
import os
import shutil


def create_data_folders():
    """데이터 폴더 생성"""
    folders = ["raw", "processed", "external"]
    for folder in folders:
        os.makedirs(f"data/{folder}", exist_ok=True)


def copy_raw_data():
    """원본 데이터 복사"""
    # source_path = "../../bike_train.csv"
    destination_path = "data/raw/bike_train.csv"
    # shutil.copy2(source_path, destination_path)


def main():
    create_data_folders()
    copy_raw_data()


if __name__ == "__main__":
    main()
