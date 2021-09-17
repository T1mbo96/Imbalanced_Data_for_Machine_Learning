import pandas as pd

from typing import Tuple
from imblearn.over_sampling import RandomOverSampler

from utils.split_x_y import split_x_y


def random_over_sampling(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_x, df_y = split_x_y(df=df, target=target)
    ros: RandomOverSampler = RandomOverSampler()

    return ros.fit_resample(df_x, df_y)
