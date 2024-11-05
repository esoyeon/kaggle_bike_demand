# 자전거 대여량 예측 ML 프로젝트

## 프로젝트 개요
이 프로젝트는 공유 자전거 대여량을 예측하는 머신러닝 모델을 개발하고 배포하기 위한 구조화된 프로젝트입니다.

### 데이터 출처
이 프로젝트는 Kaggle의 "Bike Sharing Demand" 경진대회의 데이터셋을 사용합니다:
- 대회 링크: [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)
- 데이터셋: Capital Bikeshare의 자전거 대여 기록
- 기간: 2011-2012년
- 목적: 시간대별 자전거 대여 수요 예측

### 데이터 설명
- datetime: 시간 단위 타임스탬프
- season: 계절 (1: 봄, 2: 여름, 3: 가을, 4: 겨울)
- holiday: 공휴일 여부
- workingday: 근무일 여부
- weather: 날씨 상태 (1: 맑음, 2: 흐림, 3: 가벼운 비/눈, 4: 악천후)
- temp: 온도 (섭씨)
- atemp: 체감 온도
- humidity: 습도
- windspeed: 풍속
- count: 대여된 자전거 수 (타겟 변수)

### 주요 기능
- 데이터 전처리 및 특성 엔지니어링
- 다양한 머신러닝 모델 학습 및 평가
- 하이퍼파라미터 최적화
- 모델 성능 평가 및 시각화
- 예측 파이프라인

## 프로젝트 구조
```
ml_project/
├── data/                    # 데이터 저장소
│   ├── raw/                # 원본 데이터
│   ├── processed/          # 전처리된 데이터
│   └── external/           # 외부 데이터
│
├── models/                 # 학습된 모델 저장
│   └── saved/
│
├── notebooks/             # Jupyter 노트북
│   └── exploratory/
│
├── src/                   # 소스 코드
│   ├── data/             # 데이터 처리
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   │
│   ├── features/         # 특성 엔지니어링
│   │   └── build_features.py
│   │
│   └── models/          # 모델 관련 코드
│       ├── train.py
│       ├── predict.py
│       └── evaluate.py
│
├── configs/              # 설정 파일
│   └── model_config.yaml
│
├── logs/                # 로그 파일
├── requirements.txt     # 의존성 패키지
└── README.md           # 프로젝트 문서
```

## 설치 방법
1. 저장소 클론
```bash
git clone [repository-url]
cd ml_project
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법
1. 데이터 준비
```bash
python src/data/make_dataset.py
```

2. 특성 엔지니어링
```bash
python src/features/build_features.py
```

3. 모델 학습
```bash
python src/models/train.py
```

4. 모델 평가
```bash
python src/models/evaluate.py
```

5. 예측 실행
```bash
python src/models/predict.py
```

## 주요 설정
- 모델 파라미터: `configs/model_config.yaml`
- 데이터 경로: 각 스크립트의 상단에 정의
- 로깅 설정: 각 모듈별로 독립적인 로그 파일 생성

## 모델 성능
- 평가 지표: RMSE, R², MAE, MAPE
- 결과 시각화: `logs/` 디렉토리에 저장
- 상세 성능 분석: `src/models/evaluate.py` 실행 결과 참조

## 참고사항
- 모든 경로는 프로젝트 루트 디렉토리 기준
- 로그 파일은 자동으로 타임스탬프 포함하여 생성
- 모델 파일은 자동으로 버전 관리됨

