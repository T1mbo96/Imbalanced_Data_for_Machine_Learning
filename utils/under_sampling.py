import pandas as pd

from typing import Tuple
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from utils.split_x_y import split_x_y


def random_under_sampling(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_x, df_y = split_x_y(df=df, target=target)
    rus: RandomUnderSampler = RandomUnderSampler()

    return rus.fit_resample(df_x, df_y)


def near_miss_under_sampling(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_x, df_y = split_x_y(df=df, target=target)
    nm: NearMiss = NearMiss()

    return nm.fit_resample(df_x, df_y)
