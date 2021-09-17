import pandas as pd

from typing import Tuple, List
from imblearn.over_sampling import SMOTENC

from utils.split_x_y import split_x_y


def smote_nc(df: pd.DataFrame, target: str, categorical_features: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_x, df_y = split_x_y(df=df, target=target)
    snc: SMOTENC = SMOTENC(categorical_features=categorical_features, k_neighbors=1, n_jobs=-1)

    return snc.fit_resample(df_x, df_y)
