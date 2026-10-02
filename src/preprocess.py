"""
preprocess.py
=============
배터리 셀 데이터셋 전처리 및 누수 방지(Hold-out) 배치 분할 모듈
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(filepath="df_clean_129.pkl"):
    """
    정제된 129개 배터리 셀 데이터셋(df_clean_129.pkl)을 로드합니다.
    파일이 없을 경우 상위 디렉토리 및 상대 경로를 자동 탐색합니다.
    """
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
    else:
        alt_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alt_path):
            df = pd.read_pickle(alt_path)
        else:
            raise FileNotFoundError(f"데이터셋 파일({filepath})을 찾을 수 없습니다.")
    
    print(f"✅ 데이터셋 로드 완료: 총 {len(df)}개 유효 셀")
    return df


def split_batches(df, test_size=0.2, random_state=42):
    """
    Severson et al. (2019) 및 노션 평가 기준에 맞게 배치를 분할합니다:
    - Batch 1 (46개): Train (80%, 36개) / Valid Hold-out (20%, 10개)
      * Hold-out 분할 채택 근거: 동일 충전 프로토콜 셀 간 데이터 누수(Data Leakage) 차단
    - Batch 2 (39개): 1차 최종 테스트셋 (초고속 충전 중심)
    - Batch 3 (44개): 2차 외부 벤치마크 추가 테스트셋
    """
    b1_all = df[df["batch_name"] == "batch1"].copy()
    b2_test = df[df["batch_name"] == "batch2"].copy()
    b3_test = df[df["batch_name"] == "batch3"].copy()

    # Batch 1 내 Hold-out 분할
    b1_train, b1_val = train_test_split(
        b1_all, test_size=test_size, random_state=random_state
    )

    print("=" * 65)
    print("           🛡️ [배터리 데이터셋 배치 분할 현황]")
    print("=" * 65)
    print(f"• Train (Batch 1 Train)    : {len(b1_train)}개 셀 (학습용)")
    print(f"• Valid (Batch 1 Hold-out) : {len(b1_val)}개 셀 (누수 없는 내부 검증용)")
    print(f"• Test (Batch 2)           : {len(b2_test)}개 셀 (1차 최종 테스트셋)")
    print(f"• Test (Batch 3 추가검증)  : {len(b3_test)}개 셀 (외부 2차 벤치마크 테스트셋)")
    print(f"• 총 유효 셀 수            : {len(df)}개 (10개 미도달 결측 셀 제외)")
    print("=" * 65)

    return b1_train, b1_val, b2_test, b3_test
