import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Logger.LoggingGenerator import Logger
import os

logger = Logger(os.path.basename(__file__).split('.')[0]).get_logger()


def jc_advisor_status_features(dataframe):
    logger.debug("Processing advisor meeting data.")
    imputing_cond = dataframe['in_contact_with_job_advisor'].isin([np.nan,'03.07.2024'])
    in_contact_ser = dataframe['in_contact_with_job_advisor']
    dataframe['feat_in_contact_with_job_advisor'] = np.where(imputing_cond, 'not specified', in_contact_ser)
    
    dataframe['feat_registered_with_the_jobcenter'] = dataframe['registered_with_the_jobcenter'].apply(
        preprocess_registered_with_the_jobcenter)
    
    return dataframe


def preprocess_visa_status(dataframe):
    def normalize_status(current_status):
        if pd.isna(current_status):
            return ":unknown"
        
        working_status = str(current_status).lower().split("(")[0]
        if working_status.find("eu citi") > -1:
            return "eu citizen"
        if working_status.find("ständiger wohnsitz") > -1 or working_status.find("permanent resident") > -1:
            return "permanent resident"
        if working_status.find("eu blue card") > -1 or working_status.find("blaue karte") > -1:
            return "eu blue card"
        if working_status.find("24") > -1:
            return "paragraph 24"
        if working_status.find("asylum") > -1 or working_status.find("24") > -1:
            return "asylum seeker"
        if working_status.find("temporary stay") > -1 or working_status.find("aufenthaltserlaubnis") > -1:
            return "temporary stay permit"
        
        if working_status.find("work visa") > -1:
            return "work visa"
        
        if working_status.find("student visa") > -1:
            return "student visa"
        if working_status.find("famil") > -1:
            return "family reunification"
        
        if working_status.find("job seeker") > -1:
            return "job seeker visa"
        
        if working_status.find("other") > -1:
            return ":other"
        return ":other"
    
    dataframe['feat_visa_status'] = dataframe['status'].apply(normalize_status)
    
    return dataframe


def preprocess_age_ranges(dataframe):
    dataframe['age_range'] = dataframe['age_range'].fillna("not specified")
    dataframe = dataframe.dropna(subset=["age_range"]).copy()
    dataframe['age_range'] = np.where(dataframe['age_range'].str.contains(';'), '22-24', dataframe['age_range'])
    # dataframe = dataframe[~dataframe['age_range'].str.contains(";")].copy()
    
    dataframe['age_range'] = dataframe['age_range'].replace({"32": "25-40"})
    dataframe['age_range'] = np.where(dataframe['age_range'] == '16-22', "16-21", dataframe['age_range'])
    
    dataframe['feat_age_range'] = dataframe['age_range']
    
    return dataframe


def norm_employment(status):
    status_str = str(status).lower().strip()
    unemployed_cond = [status_str.find("unempl") > -1 ]
    long_break_cond = [status_str.find("wife") > -1
            , status_str.find("leave") > -1, status_str.find("ausfrau") > -1,
                       status_str.find('elternzeit') > -1, status_str.find("home") > -1, status_str.find('mom') > -1,
                       status_str.find('mother') > -1, ]
    if any(unemployed_cond):
        return "unemployed"
    if any(long_break_cond):
        return "long_break"
    if status_str.find("self") > -1 or status_str.find("freel") > -1 or status_str.find("selbstst") > -1:
        return "freelance"
    if status_str.find("employed") > -1 or status_str.find('pair') > -1 or status_str.find("worker") > -1:
        return "employed"
    if status_str.find("train") > -1 or status_str.find("azub") > -1 or status_str.find("appre") > -1:
        return "trainee"
    if status_str.find("mini") > -1 or status_str.find("part") > -1 or status_str.find("intern") > -1:
        return "trainee"
    if status_str.find("student") > -1:
        return "student"
    if status_str.find("refugee") > -1 or status_str.find("asyl") > -1 or status_str.find("24") > -1:
        return "refugee"
    if status_str.find("usbildung") > -1:
        return "training"
    if status_str.find('retired') > -1:
        return 'retired'
    if status_str == "sonstiges" or status_str == "other":
        return "other"
    return "other"


def clean_preferred_cohort(cohort_series):
    # Standardize month names using mapping
    month_mapping = {
            "januar": "January", "februar": "February", "märz": "March", "mai": "May",
            "juni": "June", "juli": "July", "oktober": "October", "dezember": "December",
            "jan ": "January ", "feb ": "February ", "mar ": "March ", "apr ": "April ",
            "jun ": "June ", "jul ": "July ", "aug ": "August ", "sep ": "September ",
            "oct ": "October ", "nov ": "November ", "dec ": "December "
    }
    
    # Clean text & apply mapping
    cohort_series = cohort_series.str.lower().replace(month_mapping, regex=True)
    cohort_series = cohort_series.str.replace(r' class', '', regex=True).str.strip()
    
    # Convert to datetime with coercion for invalid values
    return pd.to_datetime(cohort_series, format='%B %Y', errors='coerce').dt.tz_localize('UTC')


def preprocess_registered_with_the_jobcenter(val):
    if pd.isna(val):
        return "not specified"
    if val.lower().strip() == "yes":
        return "yes"
    return "not yet"


def field_of_interest_preprocess(field):
    if pd.isna(field):
        return "not specified"

    # Clean text and map known patterns
    field = field.strip().lower()

    field_mapping = {
        "orientation course": ["oc", "orient", "tech"],
        "ai engineering": ["ai"],
        "cybersecurity": ["cyber", "cy"],
        "data analytics": ["data", "da"],
        "software engineering": ["software", "se", "swe"],
        "web development": ["web"],
        "cloud": ["cloud"],
        "backend": ["backend"],
        "sales": ["sales"],
        "chip design": ["chip des"],
        "it support": ["it"],
        "qa engineering": ["qa"],
        "online marketing": ["online marketing"],
        "not sure": ["i don't know yet", "not sure yet", "i'm not sure yet"]
    }

    # Efficient mapping logic
    for key, patterns in field_mapping.items():
        if any(pattern in field for pattern in patterns):
            return key

    return "other"

