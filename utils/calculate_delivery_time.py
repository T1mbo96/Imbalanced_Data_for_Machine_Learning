import numpy as np

from datetime import datetime


def calculate_delivery_time(order_date: str, delivery_date: str) -> str:
    if order_date is np.nan or delivery_date is np.nan:
        return 'Missing'

    order_date = datetime.strptime(order_date, '%Y-%m-%d').date()
    delivery_date = datetime.strptime(delivery_date, '%Y-%m-%d').date()
    delivery_time = (delivery_date - order_date).days

    if delivery_time < 0:
        return 'Missing'

    return str(delivery_time)
