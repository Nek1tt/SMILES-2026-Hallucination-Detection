from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y,
    )
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=relative_val,
        random_state=random_state, stratify=y[idx_trainval],
    )
    return [(idx_train, idx_val, idx_test)]
