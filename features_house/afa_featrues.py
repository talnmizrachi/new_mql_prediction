import re
import pandas as pd


def agent_changer(agent):
    if str(agent) in (".", "tbt", "?", "no name", "n.a.", "tbd", "pending", "/", "a/a", '-') or pd.isna(agent):
        return "unknown"
    if agent == "jc" or agent == "jobcenter":
        return "jc"
    return agent.lower().strip()


# df['feat_Agent_of_AfA']

def agent_gender(agent):
    agent = agent.lower()
    
    if "frau " in agent:
        return "female"
    if len(re.findall(r'her{1,2} ', agent)) > 0 or agent == "matthias höinghaus":
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
    if 'unknown' in (feat_gender, feat_agent_gender):
        return 'unable to determine'
    
    if feat_gender == feat_agent_gender:
        return "same gender"
    return "not the same"