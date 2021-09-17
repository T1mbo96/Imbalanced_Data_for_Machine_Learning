import pandas as pd

from typing import Tuple


def split_x_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df.drop(target, axis=1), df.loc[:, [target]]
