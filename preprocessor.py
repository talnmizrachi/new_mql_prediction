import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from features_house.afa_featrues import *
from features_house.constants_and_maps import language_mapping_dict, education_mapping, location_mapping, \
    work_experience_mapping, states_to_continent_map
from features_house.marketing_features import preprocess_utm_source, preprocess_utm_medium, \
    preprocess_datetime_columns_hour_minutes_features
from features_house.ms_features import create_success_dict
from features_house.states_features import generate_states_data
from features_house.student_features import preprocess_age_ranges, norm_employment, jc_advisor_status_features, \
    clean_preferred_cohort, preprocess_visa_status, field_of_interest_preprocess
from features_house.text_features import extract_all_text_features, preprocess_text
from labels_house.labels_functions import multiply_plans_by_value
from Logger.LoggingGenerator import Logger
import os

logger = Logger(os.path.basename(__file__).split('.')[0]).get_logger()


def other_candidate_features(_df):
    logger.info("Processing preferred cohort and expectation features.")
    
    # Clean and transform 'preferred_cohort'
    _df['preferred_cohort'] = clean_preferred_cohort(_df['preferred_cohort'])
    
    # Expectation Feature
    _df['feat_expectation'] = (_df['preferred_cohort'] - _df['mql_date']).dt.days
    _df['feat_expectation'].fillna(_df['feat_expectation'].mean(), inplace=True)
    _df['feat_expectation'] = np.sign(_df['feat_expectation']) * np.log1p(np.abs(_df['feat_expectation']))
    
    # Field of Interest Feature
    _df['feat_field_of_interest'] = _df['field_of_interest'].fillna('not specified').apply(field_of_interest_preprocess)
    
    # Residents Feature
    _df['feat_residents'] = _df['residents'].fillna("not specified")
    
    # Days to MQL Feature
    _df['feat_days_to_mql'] = np.log1p((_df['mql_date'] - _df['createdate']).dt.days)
    
    return _df


def past_experience_features(_df):
    logger.debug("Correcting and mapping last job details.")
    _df["last_job"] = _df["last_job"].str.replace("Hotal/Tourism", "Hotel/Tourism").str.lower().fillna("not specified")
    _df['feat_last_job_related'] = _df['last_job'].isin(["it", "it branche", "web development", "marketing"])
    work_experience_map = work_experience_mapping()
    _df["feat_work_experience_category"] = _df["time_worked_in_de"].fillna("not specified").map(work_experience_map)
    return _df


def get_labels(_df):
    logger.info("Creating enrollment and regression labels.")
    _df['category_label_bg_enrolled'] = _df['closed_won_deal__program_duration'].replace("7 Months", "8 Months").fillna(
        "not enrolled")

    _df['regression_label'] = _df['closed_won_deal__program_duration'].apply(multiply_plans_by_value)
    _df['got_sql'] = np.where(_df['sql_date'].notna(), "sql", "no_sql")
    _df['enrolled'] = np.where(_df['closed_won_deal__program_duration'].notna(), "enrolled", "not_enrolled")
    _df['label_potential_high'] = np.where(_df['deal_potential'].str.lower()=='high', 'high', "not high")
    _df['label_potential_medium'] = np.where(_df['deal_potential'].str.lower() == 'medium', 'medium', "not medium")
    _df['label_potential_low'] = np.where(_df['deal_potential'].str.lower() == 'low', 'low', "not low")
    _df['label_deal_potential'] = _df['deal_potential'].str.lower().fillna("unknown")
    
    #requested_bg - requested/not requested
    return _df


def get_text_tfidf_features(_df, max_features=2500, n_components=150, verbose=True, save_files=True):

    if save_files:
        _df.to_pickle(f"interim_datasets/df_tokenized_version_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl")

    # Pipeline with TfidfVectorizer and TruncatedSVD
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,

            ngram_range=(1, 5),
            max_df=0.85,
            min_df=3,
        )),
        ('svd', TruncatedSVD(
            n_components=min(n_components, max_features),
            random_state=42
        ))
    ])

    # Fit and transform the data
    X_reduced = text_pipeline.fit_transform(_df['tokenized_version'].values).astype('float32')


    # Create SVD columns
    svd_columns = [f'svd_component_{i + 1}' for i in range(X_reduced.shape[1])]
    svd_df = pd.DataFrame(X_reduced, index=_df.index, columns=svd_columns)

    # Combine the DataFrames
    df_combined = pd.concat([_df, svd_df], axis=1)

    # Save pipeline instead of separate vectorizer
    if save_files:
        with open(f'pickle_jar/text_pipeline_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
            pickle.dump(text_pipeline, f)

    if verbose:
        print(f"TF-IDF + SVD Pipeline Output Shape: {X_reduced.shape}")

    return df_combined, text_pipeline


def deals_related_features(_df, _deals_file):
    logger.debug("Processing deals data and joining with main DataFrame.")
    deals = pd.read_csv(_deals_file, sep="\t", dtype={"hubspot_id ": str})

    deals['hubspot_id'] = deals['hubspot_id'].astype(np.int64)
    deals = deals.set_index("hubspot_id")[['deal_potential', 'agent_of_afa', 'deal_owner']].copy()
    _df = _df.join(deals.dropna(), how="left")
    _df['feat_Deal_owner'] = _df['deal_owner'].fillna("unknown")
    _df['feat_Agent_of_AfA'] = _df['agent_of_afa'].str.lower().fillna("unknown")
    _df['feat_Deal_potential'] = _df['deal_potential'].fillna("unknown")
    
    return _df


def agent_candidate_relations_features(_df):
    logger.info("Generating additional agent gender features.")
    
    _df['feat_Agent_of_AfA'] = _df['feat_Agent_of_AfA'].apply(agent_changer)
    _df["feat_agent_gender"] = _df['feat_Agent_of_AfA'].apply(agent_gender)
    _df['feat_same_gender'] = _df[['feat_gender', "feat_agent_gender"]].apply(point_out_genders, axis=1)
    _df['feat_agent_is_known'] = _df['feat_Agent_of_AfA'].apply(agent_is_known)
    return _df


def inferring_features_from_names(_df, _names_file):
    logger.debug("Joining name and location data.")
    names_locations_genders = pd.read_csv(_names_file)
    names_locations_genders['1st Country'] = names_locations_genders['1st Country'].fillna("unable_to_identify")
    names_locations_genders = names_locations_genders.set_index("id")
    joined_df_with_locations = _df.join(names_locations_genders, how='left')
    joined_df_with_locations["feat_name_provided"] = np.where(joined_df_with_locations['1st Country'].isna(),
                                                              "name_not_provided", "provided_name")
    joined_df_with_locations['feat_gender'] = joined_df_with_locations['Gender'].fillna("unknown")
    country_to_continent = states_to_continent_map()
    joined_df_with_locations['feat_1st_country'] = joined_df_with_locations['1st Country'].fillna("unknown")
    joined_df_with_locations['feat_continent'] = joined_df_with_locations['feat_1st_country'].map(country_to_continent)
    _df = joined_df_with_locations.copy()
    
    return _df


def geographical_features(_df):
    logger.info("Merging location and state data.")
    location_map_dict = location_mapping()
    _df['Location'] = _df['state'].fillna("not specified").map(location_map_dict)
    states_data = generate_states_data()
    merged_df = _df.merge(states_data, on='Location', how="left")
    merged_df.index = _df.index
    merged_df = merged_df.rename({"Location": "feat_state_in_de"}, axis=1)
    merged_df['feat_state_in_de'] = merged_df['feat_state_in_de'].str.lower()
    
    logger.debug("Calculating success ratios per state.")
    ratios = merged_df[~merged_df['feat_state_in_de'].isna()][['feat_state_in_de', 'requested_bg']].pivot_table(
        index="feat_state_in_de", columns="requested_bg", aggfunc=len, fill_value=0, margins=True
    )
    merged_df['feat_past_success_for_ms'] = merged_df['feat_state_in_de'].map(
        (100 * ratios['requested'] / ratios['All']).to_dict()
    )
    _df = merged_df.copy()
    
    logger.info("Filling missing unemployment and industry demand rates.")
    _df['feat_unemployment_rate_pct'] = _df['feat_unemployment_rate_pct'].fillna(
        _df['feat_unemployment_rate_pct'].median())
    _df['feat_industry_demand_pct'] = _df['feat_industry_demand_pct'].fillna(_df['feat_industry_demand_pct'].median())
    
    logger.debug("Applying random imputation for state specific funding policies.")
    _df['feat_state_specific_funding_policies'] = _df['feat_state_specific_funding_policies'].apply(
        lambda x: np.random.choice([True, False], p=[0.388176, 0.611824]) if pd.isna(x) else x
    )
    
    logger.info("Creating month proportion feature and east-west feature.")
    month_proportion_dict = create_success_dict(_df[['mql_date', 'requested_bg']])
    _df['feat_requesting_prop'] = _df['mql_date'].dt.month.map(month_proportion_dict)
    _df['feat_east_west'] = _df['feat_east_west'].fillna("unknown")
    
    logger.info("Finalizing main DataFrame after merging state data.")
    
    return _df


def marketing_related_features(_df):
    campaign_types = pd.read_csv('campaign_types.tsv', sep="\t", dtype=(str,str))
    _df['feat_lpvariant'] = _df['lpvariant'].fillna("unknown")
    _df['feat_campaign_type'] = _df['utm_campaign'].astype(str).map(campaign_types.set_index('campaign_id').to_dict()['campaign_type']).fillna("ir")
    _df['feat_utm_source'] = _df['utm_source'].apply(preprocess_utm_source)
    _df['feat_utm_campaign'] = _df['utm_campaign'].fillna("unknown")
    _df['feat_utm_campaign_special'] = _df['utm_campaign'] == '16606269324'
    _df['feat_utm_term'] = _df['utm_term'].fillna("unknown")
    _df['feat_utm_medium'] = _df['utm_medium'].apply(preprocess_utm_medium)
    
    return _df


def education_level_features(_df):
    logger.info("Mapping education levels.")
    lower_levels = ["No Formal Education", "Secondary Education", "Vocational Training", "Other"]
    ed_map = education_mapping()
    _df["feat_education_level"] = _df["highest_level_of_education"].map(ed_map).str.lower().fillna("not specified")
    _df['feat_education_low'] = _df['highest_level_of_education'].isin(lower_levels)
    return _df


def langauge_features(_df):
    logger.debug("Mapping language levels.")
    language_mapping_dictionary = language_mapping_dict()
    _df['english_level_pat'] = _df['english_level'].str.strip().str.extract(r"\((.*?)\)")[0].fillna(
        _df['english_level'])
    _df['feat_eng_level_grouped'] = _df['english_level_pat'].map(language_mapping_dictionary)
    _df['german_level_pat'] = _df['german_level'].str.strip().str.extract(r"\((.*?)\)")[0].fillna(_df['german_level'])
    _df['feat_german_level_grouped'] = _df['german_level_pat'].str.strip().map(language_mapping_dictionary)
    _df['feat_form_language'] = _df['form_language'].fillna("not specified")
    _df['feat_bilingual'] = _df[['feat_german_level_grouped', 'feat_eng_level_grouped']].sum(axis=1) == 6
    
    return _df


def profile_completion_features(_df):
    logger.info("Feature engineering for profile completion started.")
    _df['feat_finished_completing_profile'] = ~_df['why_do_you_want_to_start_a_career_in_tech'].isna()
    started_completing_profile = (
            ~_df['last_job'].isna() |
            ~_df['time_worked_in_de'].isna() |
            ~_df['highest_level_of_education'].isna() |
            ~_df['state'].isna() |
            ~_df['how_did_you_hear_about_us'].isna() |
            ~_df['why_do_you_want_to_start_a_career_in_tech'].isna()
    )
    _df['feat_started_completing_profile'] = started_completing_profile
    _df['feat_completing_profile_rate'] = (
            _df[['last_job', 'time_worked_in_de', 'highest_level_of_education', 'state',
                 'how_did_you_hear_about_us', 'why_do_you_want_to_start_a_career_in_tech']]
            .notnull().sum(axis=1) / 6
    )
    return _df


def one_line_features(_df):
    _df['feat_employment_situation'] = _df['employment_situation'].apply(norm_employment)
    _df['feat_german_job_permit'] = _df['german_job_permit'].str.lower().fillna("not specified")
    _df['feat_desired_program_length'] = _df['program_length'].fillna("not specified")
    
    return _df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the column names by removing prefixes.
    """
    logger.debug("Starting to clean column names.")
    # First remove "bg__" prefix and split on "properties_"
    df.columns = [
            x.replace("bg__", "").split("properties_")[1] if "properties_" in x else x
            for x in df.columns
    ]
    # Then remove the "bg0__" prefix if it exists
    df.columns = [
            x.split("bg0__")[1] if "bg0__" in x else x
            for x in df.columns
    ]
    logger.info("Column names cleaned successfully.")
    return df


def convert_date_columns(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """
    Converts specified columns to datetime.
    """
    logger.debug("Converting date columns: %s", date_cols)
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception as e:
            logger.error("Error converting column '%s': %s", col, e)
    logger.info("Date columns conversion completed.")
    return df


def load_and_preprocess_data(new_data_file: str, names_file: str, deals_file: str):
    """
    Loads data from CSV files and performs full preprocessing.

    Parameters:
        new_data_file: Path to the main TSV file.
        names_file: Path to the CSV file with names and locations.
        deals_file: Path to the CSV file with deal information.

    Returns:
        A tuple of (processed DataFrame, tfidf vectorizer, tfidf matrix).
        :rtype: tuple
    """
    
    if 'df_tokenized_version_20250324_090957.pkl' not in os.listdir('interim_datasets'):
    
        logger.info("Loading data from file: %s", new_data_file)
        try:
            new_df = pd.read_csv(new_data_file, sep="\t", index_col='id')
        except Exception as e:
            logger.critical("Failed to load data from %s: %s", new_data_file, e)
            raise
        
        new_df = clean_column_names(new_df)
        
        date_columns = ['requested_bg_date', 'createdate', 'mql_date', 'sql_date', 'bg_enrolled_date']
        new_df = convert_date_columns(new_df, date_columns)
        
        logger.debug("Starting preprocessing age ranges and employment information.")
        df = preprocess_age_ranges(new_df)
        df = one_line_features(df)
        df = profile_completion_features(df)
        df = langauge_features(df)
        df = education_level_features(df)
        df = marketing_related_features(df)
        df = jc_advisor_status_features(df)
        df = geographical_features(df)
        df = preprocess_visa_status(df)
        df = preprocess_datetime_columns_hour_minutes_features(df, 'mql_date')
        df = inferring_features_from_names(df, names_file)
        df = past_experience_features(df)
        df = other_candidate_features(df)
        
        df.to_pickle("interim_datasets/1 - joined_df_with_locations.pkl")
        df['why_do_you_want_to_start_a_career_in_tech'] = df['why_do_you_want_to_start_a_career_in_tech'].fillna("")
        df[["tokenized_version", "detected_language"]] = df['why_do_you_want_to_start_a_career_in_tech'].apply(
            lambda x: pd.Series(preprocess_text(x))
        )
    else:
        df = pd.read_pickle('interim_datasets/df_tokenized_version_20250324_090957.pkl')
        for n_com in (500,):
            df, text_pipeline = get_text_tfidf_features(df, n_components=n_com, max_features=2500, save_files=False)
            logger.debug("Extracting text features for career motivation.")
            text_grades = extract_all_text_features(df['why_do_you_want_to_start_a_career_in_tech'])
            with_grades = df.join(text_grades, how="inner")
            df = with_grades.copy()
    
    return df, text_pipeline


# Example usage:
if __name__ == '__main__':
    try:
        processed_df, tfidf_vectorizer = load_and_preprocess_data(new_data_file='new_data.tsv',
                                                                                names_file='names.csv',
                                                                                deals_file='all-deals.tsv')
        processed_df.to_csv("training_data/preprocessed_df.tsv", sep="\t", index_label="id")
        
        logger.info("Data loaded and preprocessed successfully. Here's a preview:")
        print(processed_df.head())
    except Exception as e:
        logger.critical("A critical error occurred during preprocessing: %s", e)
