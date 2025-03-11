def agent_changer(agent):
    if agent in (".", "tbt", "?","no name","n.a.", "tbd","pending","/","a/a") or "-" in agent:
        return "unknown"
    if agent=="jc" or agent=="jobcenter":
        return "jc"
    return agent
# df['feat_Agent_of_AfA']

def agent_gender(agent):
    if agent.startswith("frau"):
        return "female"
    if agent.startswith("herr"):
        return "male"
    return "unknown"


def point_out_genders(cols):
    feat_gender, feat_agent_gender = cols
    if 'unknown' in (feat_gender, feat_agent_gender):
        return 'unable to determine'
    if feat_gender == feat_agent_gender:
        return "same gender"
    return "not the same"