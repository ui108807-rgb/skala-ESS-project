"""
features.py
===========
배터리 수명 예측 4대 핵심 피처 정의 및 변환 모듈
"""

import numpy as np
import pandas as pd

# 4대 핵심 피처셋 (전압 미세 변형 + C-rate 충전 속도 스트레스 + 발열 열화 + 방전 용량 변동성)
FEATURE_NAMES = [
    "dQ_log_var",        # log10(var(dQ_100-10(V))) : 내부 활물질 손실(LAM) 및 리튬 석출(LLI) 지표
    "mean_chargetime",   # 초기 100사이클 평균 충전 시간 : 충전 프로토콜(C-rate) 스트레스 지표
    "mean_Tmax",         # 초기 100사이클 평균 최고 온도 : 급속 충전 발열 열화 지표
    "std_QD"             # 초기 100사이클 방전 용량 표준편차 : 용량 안정화(Capacity Rise) 변동성 지표
]

TARGET_NAME = "log_cycle_life"
RAW_TARGET_NAME = "cycle_life"


def extract_features_and_target(df, features=None, target=TARGET_NAME):
    """
    데이터프레임에서 지정된 피처 행렬(X)과 타깃 벡터(y)를 분리하여 반환합니다.
    """
    if features is None:
        features = FEATURE_NAMES
    
    X = df[features].copy()
    y = df[target].values if target in df.columns else None
    y_raw = df[RAW_TARGET_NAME].values if RAW_TARGET_NAME in df.columns else None
    
    return X, y, y_raw
