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
    # Standardize German months to English
    month_mapping = {
            "Januar": "January",
            "Februar": "February",
            "März": "March",
            "Mai": "May",
            "Juni": "June",
            "Juli": "July",
            "Oktober": "October",
            "Dezember": "December"
    }
    
    # Replace German month names
    cohort_series = cohort_series.replace(month_mapping, regex=True)
    
    # Remove words like "Class" and extra spaces
    cohort_series = cohort_series.str.replace(r' Class', '', regex=True).str.strip()
    
    # Fix abbreviated months (e.g., 'Jan 2025' -> 'January 2025')
    cohort_series = cohort_series.replace({
            "Jan ": "January ",
            "Feb ": "February ",
            "Mar ": "March ",
            "Apr ": "April ",
            "Jun ": "June ",
            "Jul ": "July ",
            "Aug ": "August ",
            "Sep ": "September ",
            "Oct ": "October ",
            "Nov ": "November ",
            "Dec ": "December "
    }, regex=True)
    
    # Convert to datetime, forcing errors to NaT for invalid entries
    return pd.to_datetime(cohort_series, format='%B %Y', errors='coerce').dt.tz_localize('UTC')


def preprocess_registered_with_the_jobcenter(val):
    if pd.isna(val):
        return "not specified"
    if val.lower().strip() == "yes":
        return "yes"
    return "not yet"


def field_of_intreset_preprocess(field):
    if pd.isna(field):
        return "not specified"
    field = field.strip().lower()
    if "oc" in field or "orient" in field or "tech" in field:
        return "orientation course"
    if field.startswith("ai"):
        return "ai engineering"
    if "cyber" in field or field.startswith("cy"):
        return "cybersecurity"
    if "data" in field or field.startswith("da"):
        return "data analytics"
    if "software" in field or "se" == field or "swe" in field:
        return "software engineering"
    if "web" in field:
        return "web development"
    if "cloud" in field:
        return "cloud"
    if "backend" in field:
        return "backend"
    if "sales" in field:
        return "sales"
    if "chip des" in field:
        return "chip design"
    if field.startswith("it"):
        return "it support"
    if field.startswith("qa"):
        return "qa engineering"
    if field == 'online marketing':
        return field
    if field in ("i don't know yet", "not sure yet", "i'm not sure yet"):
        return "not sure"
    
    return f"other"

