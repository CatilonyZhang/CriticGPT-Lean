from datetime import datetime


def get_datetime(readable: bool = False) -> str:
    if readable:
        return datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    return datetime.now().strftime("%Y%m%d_%H%M%S")
