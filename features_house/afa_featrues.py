import re
import pandas as pd


def agent_changer(agent):
    if str(agent) in (".", "tbt", "?", "no name", "n.a.", "tbd", "pending", "/", "a/a", '-') or pd.isna(agent):
        return "unknown"
    if agent == "jc" or agent == "jobcenter":
        return "jc"
    return agent.lower().strip()


# df['feat_Agent_of_AfA']

# Known agent exceptions for more scalable logic
KNOWN_MALE_AGENTS = {"matthias höinghaus"}
KNOWN_FEMALE_PATTERNS = [r"\bfrau\b"]
KNOWN_MALE_PATTERNS = [r"\bherr{1,2}\b"]


def agent_gender(agent):
    # Clean text for consistent processing
    agent = agent.strip().casefold()

    # Check for explicit matches
    if agent in KNOWN_MALE_AGENTS:
        return "male"

    # Search for gender markers
    if any(re.search(pattern, agent) for pattern in KNOWN_FEMALE_PATTERNS):
        return "female"
    if any(re.search(pattern, agent) for pattern in KNOWN_MALE_PATTERNS):
        return "male"

    return "unknown"


def agent_is_known(agent):
    agent = agent.lower()
    if "frau " in agent or 'herr ' in agent:
        return "known"
    if agent in ('möcking', 'adler', 'köhler', 'schulz'):
        return "known"
    return "unknown"


def point_out_genders(cols):
    feat_gender, feat_agent_gender = cols
    
    # Handle unknowns efficiently
    if 'unknown' in {feat_gender, feat_agent_gender}:
        return 'unable to determine'
    
    # Efficient comparison
    return "same gender" if feat_gender == feat_agent_gender else "not the same"