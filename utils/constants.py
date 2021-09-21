ALLOWED_DATASET_TYPES = (
    'train',
    'validation',
    'test'
)

NUMBER_TO_WORD_MAPPING = {
    1: 'one',
    3: 'three',
    5: 'five',
    25: 'twenty_five',
    50: 'fifty'
}

SPLIT_MAPPING = {
    True: 'y',
    False: 'X'
}

TECHNIQUES_MAPPING = {
    'ru': 'random_under',
    'nmu': 'near_miss_under',
    'ro': 'random_over',
    'snc': 'smote_nc',
    'tsnc': 'tome_smote_nc'
}