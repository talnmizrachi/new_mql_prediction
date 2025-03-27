import pandas as pd
import numpy as np
import re


def preprocess_utm_source(source_col):
    if pd.isna(source_col):
        return "unknown"

    # Clean text for consistency
    source_col = source_col.strip().casefold()

    # Dictionary-based mapping for clarity and scalability
    source_mapping = {
        "google": ["google", "adwords"],
        "instagram": ["ig", "instagram"],
        "facebook": ["fb", "facebook", "meta"],
        "ms_property": ["website", "webiste", "landing page", "campus"],
        "email": ["newsletter", "email", "hs_automation", "hs_email"],
        "whatsapp": ["whatsapp"]
    }

    # Search for exact matches
    for category, keywords in source_mapping.items():
        if source_col in keywords:
            return category

    # Handle specific substring conditions
    if "source" in source_col:
        return "unknown"

    return "other"


def preprocess_utm_medium(medium_col):
    if pd.isna(medium_col):
        return "unknown"

    # Clean text for consistency
    medium_col = medium_col.strip().casefold()

    # Dictionary-based mapping for clarity and scalability
    medium_mapping = {
        "paid": ["cpc", "paid", "ppc"],
        "organic": ["organic"],
        "direct": ["direct"],
        "social": ["social"],
    }

    # Search for exact matches
    for category, keywords in medium_mapping.items():
        if medium_col in keywords:
            return category

    # Special substring conditions
    if "utm_medium" in medium_col:
        return "unknown"

    # Influencer check using a set for scalability
    known_influencers = {"niklas", "leonidas", "cedric"}
    if any(name in medium_col for name in known_influencers):
        return "influencer"

    return "unknown"



def preprocess_all_data_ad_name(dataframe):
    dataframe['ad_name'] = dataframe['ad_name'].str.lower().str.strip()
    pattern = re.compile(r'\[(?P<key>type|usp|prog|3s):(?P<value>[^\]]+)\]')
    
    def normalizing_ad_type(type_):
        if pd.isna(type_):
            return "not specified"
        if "mage" in type_:
            return "image"
        if "carou" in type_:
            return "carousel"
        if "vid" in type_:
            return "video"
        return type_
    
    def program_in_ad_imputer(program):
        if pd.isna(program):
            return "not specified"
        if 'data' in program or program == 'date':
            return 'da'
        if program.startswith("_") or program.startswith(":"):
            return program[1:].replace("-", " ")
        
        return program.replace("-", " ")
    
    def extract_selected_keys(ad_name):
        """Extract type, usp, and prog from the ad_name string and return them as a dictionary."""
        matches = pattern.finditer(ad_name)
        return {m.group('key'): m.group('value').strip('_') for m in matches}
    
    # Apply function to ad_name column and create a DataFrame from the dictionaries
    extracted_df = dataframe['ad_name'].apply(lambda x: pd.Series(extract_selected_keys(x), dtype='object'))
    extracted_df['feat_usp'] = extracted_df['usp'].fillna("not specified")
    extracted_df['feat_3s'] = extracted_df['3s'].fillna("not specified")
    extracted_df['feat_type'] = extracted_df['type'].fillna("not specified").apply(normalizing_ad_type)
    extracted_df['feat_prog'] = extracted_df['prog'].fillna("not specified").apply(program_in_ad_imputer)
    # Join the new columns to the original DataFrame
    dataframe = dataframe.join(extracted_df)
    
    return dataframe


def preprocess_datetime_columns_hour_minutes_features(df, datetime_column, drop_original=True):
    """
    Creates cyclical features from a datetime column and adds a single projection feature.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the datetime column
    datetime_column : str
        Name of the datetime column
    drop_original : bool, default=True
        Whether to drop the original datetime column

    Returns:
    --------
    pandas.DataFrame
        DataFrame with cyclical features added
    """
    # Make a copy to avoid modifying the original
    df_result = df.copy()
    
    # Convert to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_result[datetime_column]):
        df_result[datetime_column] = pd.to_datetime(df_result[datetime_column])
    
    # Extract time components
    df_result['hour'] = df_result[datetime_column].dt.hour
    df_result['day'] = df_result[datetime_column].dt.day
    df_result['month'] = df_result[datetime_column].dt.month
    df_result['day_of_week'] = df_result[datetime_column].dt.dayofweek
    
    # Create cyclical features
    
    # Hour of day (24-hour cycle)
    df_result['hour_sin'] = np.sin(2 * np.pi * df_result['hour'] / 24)
    df_result['hour_cos'] = np.cos(2 * np.pi * df_result['hour'] / 24)
    df_result[f'feat_hour_projection_{datetime_column}'] = (df_result['hour_sin'] + df_result['hour_cos']) / np.sqrt(2)
    
    # Day of month (assuming 31 days max)
    df_result['day_sin'] = np.sin(2 * np.pi * df_result['day'] / 31)
    df_result['day_cos'] = np.cos(2 * np.pi * df_result['day'] / 31)
    df_result[f'feat_day_projection_{datetime_column}'] = (df_result['day_sin'] + df_result['day_cos']) / np.sqrt(2)
    
    # Month of year (12-month cycle)
    df_result['month_sin'] = np.sin(2 * np.pi * df_result['month'] / 12)
    df_result['month_cos'] = np.cos(2 * np.pi * df_result['month'] / 12)
    df_result[f'feat_month_projection_{datetime_column}'] = (df_result['month_sin'] + df_result['month_cos']) / np.sqrt(
        2)
    
    # Day of week (7-day cycle)
    df_result['day_of_week_sin'] = np.sin(2 * np.pi * df_result['day_of_week'] / 7)
    df_result['day_of_week_cos'] = np.cos(2 * np.pi * df_result['day_of_week'] / 7)
    df_result[f'feat_day_of_week_projection_{datetime_column}'] = (df_result['day_of_week_sin'] + df_result[
        'day_of_week_cos']) / np.sqrt(
        2)
    
    # Drop intermediate columns if needed
    if drop_original:
        df_result = df_result.drop(['hour', 'day', 'month', 'day_of_week'], axis=1)
    
    return df_result