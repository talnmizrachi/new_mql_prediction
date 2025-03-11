import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from features_house.afa_featrues import agent_changer, agent_gender, point_out_genders
from features_house.constants_and_maps import language_mapping_dict, education_mapping, location_mapping, \
    work_experience_mapping, states_to_continent_map
from features_house.marketing_features import preprocess_utm_source, preprocess_utm_medium, \
    preprocess_mql_hour_minutes_features
from features_house.ms_features import create_success_dict
from features_house.states_features import generate_states_data
from features_house.student_features import preprocess_age_ranges, norm_employment, preprocess_is_meeting_with_advisor, \
    clean_preferred_cohort, preprocess_registered_with_the_jobcenter, preprocess_visa_status, \
    field_of_intreset_preprocess
from features_house.text_features import extract_all_text_features, preprocess_text
from labels_house.labels_functions import multiply_plans_by_value


new_df = pd.read_csv('new_data.tsv', sep="\t", index_col='id')
new_df.columns = [x.split("bg0__")[1] if "bg0__" in x else x for x in
                  [x.replace("bg__", "").split("properties_")[1] if "properties_" in x else x for x in new_df.columns]]
new_df['requested_bg_date'] = pd.to_datetime(new_df['requested_bg_date'])
new_df['createdate'] = pd.to_datetime(new_df['createdate'])
new_df['requested_bg_date'] = pd.to_datetime(new_df['requested_bg_date'])
new_df['mql_date'] = pd.to_datetime(new_df['mql_date'])
new_df['sql_date'] = pd.to_datetime(new_df['sql_date'])
new_df['bg_enrolled_date'] = pd.to_datetime(new_df['bg_enrolled_date'])

df = preprocess_age_ranges(new_df)
df['feat_employment_situation'] = df['employment_situation'].apply(norm_employment)
df['feat_german_job_permit'] = df['german_job_permit'].str.lower().fillna("not specified")

df['feat_finished_completing_profile'] = ~df['why_do_you_want_to_start_a_career_in_tech'].isna()
started_completing_profile = ~df['last_job'].isna() | ~df['time_worked_in_de'].isna() | ~df[
    'highest_level_of_education'].isna() | ~df['state'].isna() | ~df['how_did_you_hear_about_us'].isna() | ~df[
    'why_do_you_want_to_start_a_career_in_tech'].isna()
df['feat_started_completing_profile'] = started_completing_profile

df['feat_completing_profile_rate'] = df[['last_job', 'time_worked_in_de', 'highest_level_of_education', 'state',
                                         'how_did_you_hear_about_us',
                                         'why_do_you_want_to_start_a_career_in_tech']].notnull().sum(axis=1) / 6

language_mapping_dictionary = language_mapping_dict()
df['english_level_pat'] = df['english_level'].str.strip().str.extract(r"\((.*?)\)")[0].fillna(df['english_level'])
df['feat_eng_level_grouped'] = df['english_level_pat'].map(language_mapping_dictionary)
df['german_level_pat'] = df['german_level'].str.strip().str.extract(r"\((.*?)\)")[0].fillna(df['german_level'])
df['feat_german_level_grouped'] = df['german_level_pat'].str.strip().map(language_mapping_dictionary)
df['feat_form_language'] = df['form_language'].fillna("not specified")
df['feat_bilingual'] = df[['feat_german_level_grouped', 'feat_eng_level_grouped']].sum(axis=1) == 6

df["feat_education_level"] = df["highest_level_of_education"].map(education_mapping()).str.lower().fillna("not specified")
df['feat_education_low'] = df['highest_level_of_education'].isin(
    ["No Formal Education", "Secondary Education", "Vocational Training", "Other"])

df = preprocess_is_meeting_with_advisor(df)
df["last_job"] = df["last_job"].str.replace("Hotal/Tourism", "Hotel/Tourism").str.lower().fillna("not specified")
df['feat_last_job_related'] = df['last_job'].isin(["it", "it branche", "web development", "marketing"])

df['preferred_cohort'] = clean_preferred_cohort(df['preferred_cohort'])

df['feat_expectation'] = (df['preferred_cohort'] - df['mql_date']).dt.days
df['feat_expectation'] = df['feat_expectation'].fillna(df['feat_expectation'].mean())
standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
df['feat_expectation'] = min_max_scaler.fit_transform(df[['feat_expectation']])

df['feat_desired_program_length'] = df['program_length'].fillna("not specified")
df['feat_registered_with_the_jobcenter'] = df['registered_with_the_jobcenter'].apply(
    preprocess_registered_with_the_jobcenter)

df['feat_lpvariant'] = df['lpvariant'].fillna("unknown")
df['feat_utm_source'] = df['utm_source'].apply(preprocess_utm_source)
df['feat_utm_campaign'] = df['utm_campaign'].fillna("unknown")
df['feat_utm_term'] = df['utm_term'].fillna("unknown")
df['feat_utm_medium'] = df['utm_medium'].apply(preprocess_utm_medium)

location_map_dict = location_mapping()
df['Location'] = df['state'].fillna("not specified").map(location_map_dict)

states_data = generate_states_data()
merged_df = df.merge(states_data, on='Location', how="left")
merged_df.index = df.index

merged_df = merged_df.rename({"Location": "feat_state_in_de"}, axis=1)
merged_df['feat_state_in_de'] = merged_df['feat_state_in_de'].str.lower()

ratios = merged_df[~merged_df['feat_state_in_de'].isna()][['feat_state_in_de', 'requested_bg']].pivot_table(
    index="feat_state_in_de", columns="requested_bg", aggfunc=len, fill_value=0, margins=True)

merged_df['feat_past_success_for_ms'] = merged_df['feat_state_in_de'].map(
    (100 * ratios['requested'] / ratios['All']).to_dict())

df = merged_df.copy()
df = preprocess_visa_status(df)

work_experience_map = work_experience_mapping()
df["feat_work_experience_category"] = df["time_worked_in_de"].fillna("not specified").map(work_experience_map)

df['feat_field_of_interest'] = df['field_of_interest'].apply(field_of_intreset_preprocess)

df['feat_residents'] = df['residents'].fillna("not specified")
df["feat_days_to_mql"] = np.log1p((df['mql_date'] - df['createdate']).dt.days)
df = preprocess_mql_hour_minutes_features(df, 'mql_date')
names_locations_genders = pd.read_csv('names.csv')
names_locations_genders['1st Country'] = names_locations_genders['1st Country'].fillna("unable_to_identify")
names_locations_genders = names_locations_genders.set_index("id")

joined_df_with_locations = df.join(names_locations_genders, how='left')
joined_df_with_locations["feat_name_provided"] = np.where(joined_df_with_locations['1st Country'].isna(),
                                                          "name_not_provided", "provided_name")
joined_df_with_locations['feat_gender'] = joined_df_with_locations['Gender'].fillna("unknown")

country_to_continent = states_to_continent_map()
joined_df_with_locations['feat_1st_country'] = joined_df_with_locations['1st Country'].fillna("unknown")
joined_df_with_locations['feat_continent'] = joined_df_with_locations['feat_1st_country'].map(country_to_continent)

df = joined_df_with_locations.copy()

df['category_label_bg_enrolled'] = df['closed_won_deal__program_duration'].replace("7 Months", "8 Months").fillna(
    "not enrolled")
df['regression_label'] = df['closed_won_deal__program_duration'].apply(multiply_plans_by_value)

text_grades = extract_all_text_features(df['why_do_you_want_to_start_a_career_in_tech'])
with_grades = df.join(text_grades, how="inner")

df = with_grades.copy()

df['feat_unemployment_rate_pct'] = df['feat_unemployment_rate_pct'].fillna(df['feat_unemployment_rate_pct'].median())
df['feat_industry_demand_pct'] = df['feat_industry_demand_pct'].fillna(df['feat_industry_demand_pct'].median())


df['feat_state_specific_funding_policies'] = df['feat_state_specific_funding_policies'].apply(
    lambda x: np.random.choice([True, False], p=[0.388176, 0.611824]) if pd.isna(x) else x)

month_proportion_dict = create_success_dict(df[['mql_date', 'requested_bg']])
df['feat_requesting_prop'] = df['mql_date'].dt.month.map(month_proportion_dict)
df['feat_east_west'] = df['feat_east_west'].fillna("unknown")


features = df[[col for col in df.columns if col.startswith("feat")]]

deals = pd.read_csv('all-deals.csv', dtype={"Associated Contact IDs": str})
deals = deals[~deals['Associated Contact IDs'].isna()].copy()
deals = deals[~deals['Associated Contact IDs'].str.contains(";")].copy()
deals['Associated Contact IDs'] = deals['Associated Contact IDs'].astype(np.int64)
deals = deals.set_index("Associated Contact IDs")[['Deal owner', 'Agent of AfA/JC', 'Deal potential']].copy()

df = df.join(deals, how="left")
df['feat_Deal_owner'] = df['Deal owner'].fillna("unknown")
df['feat_Agent_of_AfA'] = df['Agent of AfA/JC'].str.lower().fillna("unknown")
df['feat_Deal_potential'] = df['Deal potential'].fillna("unknown")

df['feat_Agent_of_AfA'].apply(agent_changer).value_counts().head(35)
df["feat_agent_gender"] = df['feat_Agent_of_AfA'].apply(agent_gender)
# todo - date features for requesteed bg

df['why_do_you_want_to_start_a_career_in_tech'] = df['why_do_you_want_to_start_a_career_in_tech'].fillna("")
df["tokenized_version"] = df['why_do_you_want_to_start_a_career_in_tech'].apply(
    lambda x: preprocess_text(x)).str.lower()
vectorizer = TfidfVectorizer(
    max_features=1500,
    ngram_range=(1, 2)
    , max_df=0.85
    , min_df=3
)
tfidf_matrix = vectorizer.fit_transform(df['tokenized_version'])


df['feat_same_gender'] = df[['feat_gender', "feat_agent_gender"]].apply(point_out_genders, axis=1)
