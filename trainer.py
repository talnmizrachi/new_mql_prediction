import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from training_functions.balancing_data import balance_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings

from datetime import timedelta
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, recall_score, precision_score, \
    accuracy_score
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import pandas as pd
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from langdetect import detect
import re
import string
import contractions
import unicodedata
from langdetect import detect
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
import nltk

pd.set_option('display.max_columns', None)
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('training_data/preprocessed_df.tsv', sep="\t", index_col='id')
df['createdate'] = pd.to_datetime(df['createdate'])
df['requested_bg_date'] = pd.to_datetime(df['requested_bg_date'])
df['mql_date'] = pd.to_datetime(df['mql_date'])
df['sql_date'] = pd.to_datetime(df['sql_date'])
df['bg_enrolled_date'] = pd.to_datetime(df['bg_enrolled_date'])

features = [col for col in df.columns if col.startswith("feat")]
text_features = [col for col in df.columns if col.startswith("svd")]

all_features = text_features + features

non_feature_col = [col for col in df.columns if col not in all_features]

df = df[(df[non_feature_col].notna().sum(axis=1) / len(df.columns)) > 0.1].copy()

df['requested_bg'].value_counts()

sorted_funnel = {
        "mql": {
                "categorical": [
                        "feat_age_range",
                        "feat_employment_situation",
                        "feat_german_job_permit",
                        "feat_eng_level_grouped",
                        "feat_german_level_grouped",
                        "feat_form_language",
                        "feat_education_level",
                        "feat_in_contact_with_job_advisor",
                        "feat_desired_program_length",
                        "feat_registered_with_the_jobcenter",
                        "feat_lpvariant",
                        "feat_utm_source",
                        "feat_utm_campaign",
                        "feat_state_in_de",
                        "feat_state_specific_funding_policies",
                        "feat_east_west",
                        "feat_visa_status",
                        "feat_work_experience_category",
                        "feat_field_of_interest",
                        "feat_gender",
                        "feat_1st_country",
                        "feat_continent",
                        "feat_finished_completing_profile",
                        "feat_started_completing_profile",
                        "feat_bilingual",
                        "feat_education_low",
                        "feat_last_job_related",
                        "feat_utm_term",
                        "feat_utm_medium",
                        "feat_residents",
                        "feat_name_provided",
                
                ],
                "numerical": [
                                     "feat_completing_profile_rate",
                                     "feat_unemployment_rate_pct",
                                     "feat_industry_demand_pct",
                                     "feat_expectation",
                                     "feat_past_success_for_ms",
                                     "feat_days_to_mql",
                                     "feat_hour_projection",
                                     "feat_day_projection",
                                     "feat_month_projection",
                                     "feat_day_of_week_projection",
                                     "feat_text_length",
                                     "feat_word_count",
                                     "feat_unique_word_count",
                                     "feat_avg_word_length",
                                     "feat_char_count_no_spaces",
                                     "feat_word_density",
                                     "feat_unique_word_ratio",
                                     "feat_sentence_count",
                                     "feat_syllable_count",
                                     "feat_complex_word_count",
                                     "feat_syllables_per_word",
                                     "feat_words_per_sentence",
                                     "feat_flesch_reading_ease",
                                     "feat_flesch_kincaid_grade",
                                     "feat_gunning_fog",
                                     "feat_smog_index",
                                     "feat_coleman_liau_index",
                                     "feat_automated_readability_index",
                                     "feat_dale_chall_readability",
                                     "feat_requesting_prop",
                                     "feat_verb_counts",
                                     "feat_punctuations_count",
                             
                             ] + text_features,
            
        },
        "sql": {
                "categorical": [
                        "feat_age_range",
                        "feat_employment_situation",
                        "feat_german_job_permit",
                        "feat_eng_level_grouped",
                        "feat_german_level_grouped",
                        "feat_form_language",
                        "feat_education_level",
                        "feat_in_contact_with_job_advisor",
                        "feat_desired_program_length",
                        "feat_registered_with_the_jobcenter",
                        "feat_lpvariant",
                        "feat_utm_source",
                        "feat_utm_campaign",
                        "feat_state_in_de",
                        "feat_state_specific_funding_policies",
                        "feat_east_west",
                        "feat_visa_status",
                        "feat_work_experience_category",
                        "feat_field_of_interest",
                        "feat_gender",
                        "feat_1st_country",
                        "feat_continent",
                        "feat_Agent_of_AfA",
                        "feat_Deal_potential",
                        "feat_finished_completing_profile",
                        "feat_started_completing_profile",
                        "feat_bilingual",
                        "feat_education_low",
                        "feat_last_job_related",
                        "feat_utm_term",
                        "feat_utm_medium",
                        "feat_residents",
                        "feat_name_provided",
                        "feat_Deal_owner"
                ],
                "numerical": [
                                     "feat_completing_profile_rate",
                                     "feat_unemployment_rate_pct",
                                     
                                     "feat_industry_demand_pct",
                                     
                                     "feat_verb_counts",
                                     "feat_punctuations_count",
                                     "feat_expectation",
                                     "feat_past_success_for_ms",
                                     "feat_days_to_mql",
                                     "feat_hour_projection",
                                     "feat_day_projection",
                                     "feat_month_projection",
                                     "feat_day_of_week_projection",
                                     "feat_text_length",
                                     "feat_word_count",
                                     "feat_unique_word_count",
                                     "feat_avg_word_length",
                                     "feat_char_count_no_spaces",
                                     "feat_word_density",
                                     "feat_unique_word_ratio",
                                     "feat_sentence_count",
                                     "feat_syllable_count",
                                     "feat_complex_word_count",
                                     "feat_syllables_per_word",
                                     "feat_words_per_sentence",
                                     "feat_flesch_reading_ease",
                                     "feat_flesch_kincaid_grade",
                                     "feat_gunning_fog",
                                     "feat_smog_index",
                                     "feat_coleman_liau_index",
                                     "feat_automated_readability_index",
                                     "feat_requesting_prop",
                                     "feat_dale_chall_readability",
                             
                             ] + text_features,
        },
        "requested": {
                "categorical": [
                        "feat_age_range",
                        "feat_employment_situation",
                        "feat_german_job_permit",
                        "feat_completing_profile_rate",
                        "feat_eng_level_grouped",
                        "feat_german_level_grouped",
                        "feat_form_language",
                        "feat_education_level",
                        "feat_in_contact_with_job_advisor",
                        "feat_registered_with_the_jobcenter",
                        "feat_lpvariant",
                        "feat_utm_source",
                        "feat_utm_campaign",
                        "feat_state_in_de",
                        "feat_state_specific_funding_policies",
                        "feat_east_west",
                        "feat_visa_status",
                        "feat_work_experience_category",
                        "feat_field_of_interest",
                        "feat_gender",
                        "feat_1st_country",
                        "feat_continent",
                        "feat_Agent_of_AfA",
                        "feat_Deal_potential",
                        "feat_agent_gender",
                        "feat_same_gender",
                        "feat_agent_is_known",
                        "feat_finished_completing_profile",
                        "feat_started_completing_profile",
                        "feat_bilingual",
                        "feat_education_low",
                        "feat_last_job_related",
                        "feat_utm_term",
                        "feat_utm_medium",
                        "feat_residents",
                        "feat_name_provided",
                        
                        "feat_Deal_owner"
                ],
                "numerical": [
                                     "feat_unemployment_rate_pct",
                                     
                                     "feat_industry_demand_pct",
                                     
                                     "feat_past_success_for_ms",
                                     "feat_days_to_mql",
                                     "feat_hour_projection",
                                     "feat_day_projection",
                                     "feat_month_projection",
                                     "feat_day_of_week_projection",
                                     "feat_text_length",
                                     "feat_word_count",
                                     "feat_unique_word_count",
                                     "feat_avg_word_length",
                                     "feat_char_count_no_spaces",
                                     "feat_word_density",
                                     "feat_unique_word_ratio",
                                     "feat_sentence_count",
                                     "feat_syllable_count",
                                     "feat_complex_word_count",
                                     "feat_syllables_per_word",
                                     "feat_words_per_sentence",
                                     "feat_flesch_reading_ease",
                                     "feat_flesch_kincaid_grade",
                                     "feat_gunning_fog",
                                     "feat_smog_index",
                                     "feat_coleman_liau_index",
                                     "feat_automated_readability_index",
                                     "feat_dale_chall_readability",
                                     "feat_verb_counts",
                                     "feat_punctuations_count",
                                     "feat_requesting_prop",
                             ] + text_features,
                "others": [
                
                ]
        },
    
}

# 2. Create a boolean column indicating whether row is "requested"
df['is_requested'] = df['requested_bg'].eq('requested')

# MQL -> Requested

df['enrolled'] = df['closed_won_deal__program_duration'].notna()

starting_level = "mql"
target = 'requested_bg'

if starting_level == "requested":
    df = df[df['requested_bg'] == 'requested'].copy()
if starting_level == "sql":
    df = df[df['got_sql'] == 'sql'].copy()

cat_cols = sorted_funnel[starting_level]['categorical']
numerical = sorted_funnel[starting_level]['numerical']
final_columns = numerical + cat_cols + [target]

for col in cat_cols:
    df[col] = df[col].astype(str)

cutoff = '2024-11-01'
start_date = "2023-01-01"

text_cond = (df['why_do_you_want_to_start_a_career_in_tech'].fillna("") != "")
highest_level_of_education_is_known = (df['highest_level_of_education'].notna())
in_contact_with_job_advisor_is_known = (df['in_contact_with_job_advisor'].notna())
preferred_cohort_is_known = (df['preferred_cohort'].notna())
lpvariant_is_known = (df['lpvariant'].notna())

# conditions = text_cond & highest_level_of_education_is_known & in_contact_with_job_advisor_is_known & preferred_cohort_is_known
# conditions = text_cond & highest_level_of_education_is_known & in_contact_with_job_advisor_is_known
# conditions = text_cond & in_contact_with_job_advisor_is_known
conditions = True

future_test = df[(df['mql_date'] >= cutoff) & conditions].copy()
current_df = df[(df['mql_date'] < cutoff) & (df['mql_date'] >= start_date) & conditions].copy()

pre_model = current_df.copy()
pre_model = pre_model[final_columns].copy()
pre_model_future = future_test[final_columns].copy()

print(len(pre_model), list(pre_model[target].value_counts().items()))
print(len(pre_model_future), list(pre_model_future[target].value_counts().items()))

# highest_level_of_education
# in_contact_with_job_advisor
# preferred_cohort
# lpvariant


# Check the class distribution
print("Original class distribution:")
print(pre_model[target].value_counts())
print(f"Imbalance ratio: 1:{pre_model[target].value_counts()[0] / pre_model[target].value_counts()[1]:.2f}")

# Identify categorical and numerical columns
categorical_cols = cat_cols
numerical_cols = numerical

# Remove target from feature lists if present
if target in categorical_cols:
    categorical_cols.remove(target)
if target in numerical_cols:
    numerical_cols.remove(target)

print(f"\nCategorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# First, determine if we need to encode the target
if pre_model[target].dtype == 'object' or pre_model[target].dtype.name == 'category':
    # Encode string labels to numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(pre_model[target])
    # Check the mapping
    for i, label in enumerate(label_encoder.classes_):
        print(f"Original: '{label}' → Encoded: {i}")
else:
    # Target is already numeric
    y_encoded = pre_model[target].values

# Split the data
X = pre_model.drop(target, axis=1)
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nTrain set class distribution:")
print(Counter(y_train_encoded))

X_balanced, y_balanced, weights = balance_dataset(X_train, y_train_encoded, method="over_under", return_class_weights=True)

# Create CatBoost pools with categorical features
train_pool = Pool(X_balanced, y_balanced, cat_features=categorical_cols)
test_pool = Pool(X_test, y_test_encoded, cat_features=categorical_cols)

# Approach 1: Use balanced dataset without class weights in model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.02,
    depth=3,
    l2_leaf_reg=12,
    bootstrap_type='Bernoulli',
    subsample=0.35,
    sampling_frequency='PerTreeLevel',
    eval_metric='Precision',
    custom_metric=['F1', 'AUC'],
    loss_function='Logloss',
    class_weights=weights,
    use_best_model=True,
    random_seed=42,
    verbose=250,
    random_strength=5.15,
    od_type='Iter',
    od_wait=500,
)

model.fit(train_pool, eval_set=test_pool, plot=True, use_best_model=True)

# Evaluate the model
y_pred = model.predict(test_pool)
y_pred_proba = model.predict_proba(test_pool)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test_encoded, y_pred_proba):.4f}")

conf_matrix = confusion_matrix(y_test_encoded, y_pred)

# Get unique class labels
labels = np.unique(label_encoder.inverse_transform(y_pred))

# Plot confusion matrix with labels
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
feature_importance = model.get_feature_importance(train_pool)
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=True).reset_index(drop=True)
print("\nTop 10 Important Features:")
importance_df.head(5)

import matplotlib.pyplot as plt

# Get feature importances using the default method (PredictionValuesChange)
feature_importance = model.get_feature_importance(train_pool)
features = X_train.columns

# Visualize the importances
plt.figure(figsize=(10, 10))
plt.barh(features, feature_importance)
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance")
plt.show()

preds = model.predict(pre_model_future.drop('requested_bg', axis=1))

conf_matrix = confusion_matrix(pre_model_future[target], label_encoder.inverse_transform(preds))
print(classification_report(pre_model_future[target], label_encoder.inverse_transform(preds)))

# Get unique class labels
labels = np.unique(label_encoder.inverse_transform(preds))

# Plot confusion matrix with labels
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
