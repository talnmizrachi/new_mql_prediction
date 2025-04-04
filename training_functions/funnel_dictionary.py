def get_sorted_funnel_for_(level, text_features):
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
				"feat_campaign_type",
				"feat_utm_campaign_special",
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
				             "feat_hour_projection_mql_date",
				             "feat_day_projection_mql_date",
				             "feat_month_projection_mql_date",
				             "feat_day_of_week_projection_mql_date",
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
				"feat_campaign_type",
				"feat_utm_campaign_special",
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
				             "feat_hour_projection_mql_date",
				             "feat_day_projection_mql_date",
				             "feat_month_projection_mql_date",
				             "feat_day_of_week_projection_mql_date",
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
				"feat_campaign_type",
				"feat_utm_campaign_special",
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
				             "feat_hour_projection_mql_date",
				             "feat_day_projection_mql_date",
				             "feat_month_projection_mql_date",
				             "feat_day_of_week_projection_mql_date",
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

	return sorted_funnel[level]['categorical'], sorted_funnel[level]['numerical']
