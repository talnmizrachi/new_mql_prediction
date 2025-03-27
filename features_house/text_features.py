import nltk
import numpy as np
import pandas as pd
import math
import unicodedata
import re
import string
from langdetect import detect
import contractions
import spacy
import emoji
from nltk.corpus import stopwords
import emoji  # Optional for better emoji coverage


def check_and_download_punkt_tab():
    resource = 'tokenizers/punkt/PY3_tab'
    try:
        nltk.data.find(resource)
    except LookupError:
        print(f"Resource '{resource}' not found. Downloading now...")
        nltk.download('punkt_tab')
        print("Download complete.")


def check_and_download_spacy_models():
    try:
        nlp_en = spacy.load("en_core_web_trf")
    except Exception as e:
        from spacy.cli import download
        download("en_core_web_trf")
        nlp_en = spacy.load("en_core_web_trf")

    try:
        nlp_de = spacy.load("de_dep_news_trf")
    except Exception as e:
        from spacy.cli import download
        download("de_dep_news_trf")
        nlp_de = spacy.load("de_dep_news_trf")

    return nlp_en, nlp_de


check_and_download_punkt_tab()
nlp_en, nlp_de = check_and_download_spacy_models()


def extract_all_text_features(text_series, lang='en'):
    """
    Extract comprehensive text features including both basic metrics and advanced readability scores.

    Parameters:
    -----------
    text_series : pandas.Series()
        Series containing text data
    lang : str, default='en'
        Language model to use ('en' or 'de')

    Returns:
    --------
    pandas.DataFrame()
        DataFrame with all text features
    """
    # Initialize the DataFrame to store all features
    features_df = pd.DataFrame(index=text_series.index)
    
    # Load appropriate spaCy model
    try:
        if lang == 'en':
            nlp = spacy.load("en_core_web_trf")
        elif lang == 'de':
            nlp = spacy.load("de_dep_news_trf")
        else:
            nlp = spacy.load("en_core_web_trf")
    except OSError:
        print(f"spaCy model for language '{lang}' not found. Installing fallback model...")
        from spacy.cli import download
        try:
            download("en_core_web_trf")
            nlp = spacy.load("en_core_web_trf")
        except:
            print("Failed to download model. Using blank model instead.")
            nlp = spacy.blank("en")
    
    text_series = text_series.fillna("")
    # Basic text features
    features_df['feat_text_length'] = text_series.str.len().fillna(0)
    features_df['feat_word_count'] = text_series.apply(lambda x: len(str(x).split()) if len(x) > 2 else 0)
    features_df['feat_unique_word_count'] = text_series.apply(lambda x: len(set(str(x).split())) if len(x) > 2 else 0)
    features_df['feat_avg_word_length'] = text_series.apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 1 else 0
    )
    
    # Character-level features
    features_df['feat_char_count_no_spaces'] = text_series.apply(
        lambda x: sum(1 for char in str(x) if char not in [' ', '\t', '\n'])
    )
    
    # Additional basic features
    features_df['feat_word_density'] = features_df['feat_word_count'] / features_df['feat_text_length'].replace(0, 1)
    features_df['feat_unique_word_ratio'] = features_df['feat_unique_word_count'] / features_df[
        'feat_word_count'].replace(0, 1)
    
    features_df['feat_char_count_no_spaces'] = np.log1p(features_df['feat_char_count_no_spaces'])
    features_df['feat_avg_word_length'] = np.log1p(features_df['feat_avg_word_length'])
    features_df['feat_unique_word_count'] = np.log1p(features_df['feat_unique_word_count'])
    features_df['feat_text_length'] = np.log1p(features_df['feat_text_length'])
    features_df['feat_word_count'] = np.log1p(features_df['feat_word_count'])
    # Process each text for advanced readability metrics
    for idx, text in text_series.items():
        # Skip empty texts
        if not text or not isinstance(text, str) or text.strip() == "":
            continue
        
        # Process with spaCy
        doc = nlp(text)
        
        # Count words, sentences, syllables, characters, and complex words
        word_count = sum(1 for token in doc if not token.is_punct and not token.is_space)
        sentence_count = len(list(doc.sents))
        punctuations_count = sum(1 for token in doc if token.is_punct)
        if punctuations_count is None or pd.isna(punctuations_count):
            print(11)
        # Skip if no words or sentences
        if word_count == 0 or sentence_count == 0:
            continue
        
        # Count syllables and complex words
        syllable_count = 0
        complex_word_count = 0  # Words with 3+ syllables
        verb_counts = 0
        
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if token.pos_ == 'VERB':
                verb_counts += 1
            
            word = token.text.lower()
            
            # Count syllables (English-specific approximation)
            if lang == 'en':
                # Remove non-alphabetic characters
                word = re.sub(r'[^a-zA-Z]', '', word)
                
                # Count syllables based on vowel sequences
                count = 0
                vowels = "aeiouy"
                
                # Special cases
                if word.endswith('e'):
                    word = word[:-1]
                
                if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                    count += 1
                
                # Count vowel sequences
                prev_is_vowel = False
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_is_vowel:
                        count += 1
                    prev_is_vowel = is_vowel
                
                # Ensure at least one syllable per word
                if count == 0 and len(word) > 0:
                    count = 1
                
                syllable_count += count
                
                # Complex words (3+ syllables)
                if count >= 3:
                    complex_word_count += 1
            else:
                # For non-English languages, use a simpler approximation
                syllable_count += max(1, len(re.findall(r'[aeiouäöüáéíóúàèìòù]+', word)))
                if len(re.findall(r'[aeiouäöüáéíóúàèìòù]+', word)) >= 3:
                    complex_word_count += 1
        
        # Character count (excluding spaces)
        char_count = sum(len(token.text) for token in doc if not token.is_space)
        
        # Add sentence count and syllable count as features
        features_df.loc[idx, 'feat_sentence_count'] = sentence_count
        features_df.loc[idx, 'feat_syllable_count'] = syllable_count
        features_df.loc[idx, "feat_verb_counts"] = verb_counts
        features_df.loc[idx, 'feat_punctuations_count'] = punctuations_count
        features_df.loc[idx, 'feat_complex_word_count'] = complex_word_count
        features_df.loc[idx, 'feat_syllables_per_word'] = syllable_count / word_count
        features_df.loc[idx, 'feat_words_per_sentence'] = word_count / sentence_count
        
        # Calculate average values
        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count
        chars_per_word = char_count / word_count
        
        # Calculate readability scores
        
        # 1. Flesch Reading Ease Score
        flesch_reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        flesch_reading_ease = max(0, min(100, flesch_reading_ease))  # Clamp to 0-100
        features_df.loc[idx, 'feat_flesch_reading_ease'] = round(flesch_reading_ease, 2)
        
        # 2. Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
        flesch_kincaid_grade = max(0, flesch_kincaid_grade)  # Ensure non-negative
        features_df.loc[idx, 'feat_flesch_kincaid_grade'] = round(flesch_kincaid_grade, 2)
        
        # 3. Gunning Fog Index
        gunning_fog = 0.4 * (words_per_sentence + 100 * (complex_word_count / word_count))
        features_df.loc[idx, 'feat_gunning_fog'] = round(gunning_fog, 2)
        
        # 4. SMOG Index
        if sentence_count >= 30:
            smog_index = 1.043 * math.sqrt(complex_word_count * (30 / sentence_count)) + 3.1291
        else:
            # Adjusted formula for texts with fewer than 30 sentences
            smog_index = 1.043 * math.sqrt(complex_word_count * (30 / max(1, sentence_count))) + 3.1291
        features_df.loc[idx, 'feat_smog_index'] = round(smog_index, 2)
        
        # 5. Coleman-Liau Index
        l = (char_count / word_count) * 100  # Average number of characters per 100 words
        s = (sentence_count / word_count) * 100  # Average number of sentences per 100 words
        coleman_liau_index = max(0.0588 * l - 0.296 * s - 15.8, 0)
        features_df.loc[idx, 'feat_coleman_liau_index'] = round(coleman_liau_index, 2)
        
        # 6. Automated Readability Index (ARI)
        automated_readability_index = 4.71 * chars_per_word + 0.5 * words_per_sentence - 21.43
        automated_readability_index = max(0, automated_readability_index)  # Ensure non-negative
        features_df.loc[idx, 'feat_automated_readability_index'] = round(automated_readability_index, 2)
        
        # 7. Dale-Chall Readability Formula (simplified version)
        dale_chall_readability = 0.1579 * (complex_word_count / word_count * 100) + 0.0496 * words_per_sentence
        if complex_word_count / word_count > 0.05:
            dale_chall_readability += 3.6365  # Adjustment for texts with many complex words
        features_df.loc[idx, 'feat_dale_chall_readability'] = round(dale_chall_readability, 2)
    
    # Fill NaN values with appropriate defaults
    readability_columns = [
            'feat_flesch_reading_ease', 'feat_flesch_kincaid_grade', 'feat_gunning_fog', 'feat_smog_index',
            'feat_coleman_liau_index', 'feat_automated_readability_index', 'feat_dale_chall_readability',
            'feat_sentence_count', 'feat_syllable_count', 'feat_complex_word_count', 'feat_syllables_per_word',
            'feat_words_per_sentence'
    ]
    
    for col in readability_columns:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(0)
    
    features_df['feat_punctuations_count'] = np.log1p(features_df['feat_punctuations_count'])
    features_df['feat_verb_counts'] = np.log1p(features_df['feat_verb_counts'])
    features_df['feat_syllable_count'] = np.log1p(features_df['feat_syllable_count'])
    features_df['feat_complex_word_count'] = np.log1p(features_df['feat_complex_word_count'])
    return features_df.fillna(0)


# Emoji pattern
def remove_emojis(_text, replace_with=''):
    # Using `emoji` library for better coverage
    return emoji.replace_emoji(_text, replace_with)



# Remove accents
def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


# Assuming remove_emojis and remove_accents are defined somewhere
def preprocess_text(_text):
    _text = _text.lower()
    
    # Remove emojis, Remove accents, Expand contractions, Remove punctuations and digits, Remove excessive whitespaces
    _text = remove_emojis(_text)
    _text = remove_accents(_text)
    _text = contractions.fix(_text)
    _text = re.sub(r'[\d' + string.punctuation + r']+', '', _text)
    _text = re.sub(r'\s+', ' ', _text).strip()
    
    try:
        language = detect(_text)
    except Exception:
        language = 'en'
    
    # Truncate text to avoid exceeding model limits
    _text = " ".join(_text.split())
    
    # Load appropriate spaCy model based on detected language
    nlp = nlp_de if language == 'de' else nlp_en
    
    # Tokenization and lemmatization in one step using spaCy
    doc = nlp(_text)
    stop_words = set(stopwords.words('german') if language == 'de' else stopwords.words('english'))
    
    tokens = [token.lemma_.lower() for token in doc if token.text.lower() not in stop_words]
  
    return " ".join(tokens), language


if __name__ == '__main__':
    
    # Example usage
    text = "I've been to Paris! It's amazing! 🗼✨\n\nSo many places to visit..."
    print(preprocess_text(text))
