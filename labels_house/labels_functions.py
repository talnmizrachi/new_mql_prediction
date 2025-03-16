import pandas as pd


def multiply_plans_by_value(plan):
    if pd.isna(plan):
        return 0
    if plan in ("7 Months", "8 Months"):
        return 29600
    if plan == "1 Month":
        return 3900
    else:
        return 46540