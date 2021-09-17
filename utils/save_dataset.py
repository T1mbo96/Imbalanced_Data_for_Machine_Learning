import pandas as pd

from utils.constants import ALLOWED_DATASET_TYPES, NUMBER_TO_WORD_MAPPING, SPLIT_MAPPING, TECHNIQUES_MAPPING


def save_dataset(df: pd.DataFrame, dataset_type: str, balance: int = None, is_target: bool = None, technique: str = None) -> None:
    df.to_csv((
        f'../data/'
        f'{dataset_type if dataset_type in ALLOWED_DATASET_TYPES else "INVALID_DATASET_TYPE"}'
        f'{f"_{NUMBER_TO_WORD_MAPPING[balance]}" if balance else ""}'
        f'{f"_{SPLIT_MAPPING[is_target]}" if is_target in SPLIT_MAPPING else "" if is_target is None else "INVALID_TYPE_DISTINCTION"}'
        f'{f"_{TECHNIQUES_MAPPING[technique]}" if technique else ""}'
        f'.csv'
    ))
