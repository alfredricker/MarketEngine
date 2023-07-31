from datetime import datetime,timedelta

def get_recent_trading_day():
    # Get the current date
    current_date = datetime.now()

    # Keep subtracting one day until we find a trading day (exclude weekends and holidays)
    while True:
        if current_date.weekday() >= 5:  # 5 and 6 correspond to Saturday and Sunday
            current_date -= timedelta(days=1)
        else:
            trading_day = current_date
            break

    return trading_day

def get_trading_day_5_years_ago():
    # Get the current date
    current_date = datetime.now()

    # Subtract 5 years from the current date
    five_years_ago = current_date - timedelta(days=365 * 5)

    # Keep subtracting one day until we find a trading day (exclude weekends and holidays)
    while True:
        if five_years_ago.weekday() >= 5:  # 5 and 6 correspond to Saturday and Sunday
            five_years_ago -= timedelta(days=1)
        else:
            trading_day = five_years_ago
            break

    return trading_day

# Get the most recent trading day and the trading day 5 years ago
start_date = get_recent_trading_day()
end_date = get_trading_day_5_years_ago()