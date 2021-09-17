from datetime import datetime


def calculate_account_age_until_time_of_order(creation_date: str, order_date: str):
    creation_date = datetime.strptime(creation_date, '%Y-%m-%d').date()
    order_date = datetime.strptime(order_date, '%Y-%m-%d').date()

    return (order_date - creation_date).days
