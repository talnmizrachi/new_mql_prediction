import pandas as pd


def generate_states_data():
    states_data = {
        "Location": [
            "Baden-Württemberg",
            "Bayern (Bavaria)",
            "Berlin",
            "Brandenburg",
            "Bremen",
            "Hamburg",
            "Hessen (Hesse)",
            "Mecklenburg-Vorpommern",
            "Niedersachsen",
            "Nordrhein-Westfalen",
            "Rheinland-Pfalz",
            "Saarland",
            "Sachsen (Saxony)",
            "Sachsen-Anhalt",
            "Schleswig-Holstein",
            "Thüringen (Thuringia)",
            "Frankfurt (Hessen)",
            "Munich (Bayern)",
            "Aschaffenburg (Bayern)"
        ],
        "feat_unemployment_rate_pct": [
            3.7, 3.7, 9.7, 7.4, 11.1, 8.0, 5.3, 7.6, 5.5, 7.5, 5.6, 7.1, 6.0, 6.1,
            5.8, 6.5, 5.3, 3.7, 3.7
        ],
        "feat_state_specific_funding_policies": [
            False, False, True, False, False, True, False, False, False, True, False,
            False, False, False, False, False, False, False, False
        ],
        "feat_industry_demand_pct": [
            15, 18, 25, 10, 12, 20, 17, 8, 14, 19, 13, 9, 11, 9, 10, 8, 22, 23, 15
        ],
        "feat_east_west": [
            "West",  # Baden-Württemberg
            "West",  # Bayern (Bavaria)
            "East",  # Berlin (traditionally classified as East in many datasets)
            "East",  # Brandenburg
            "West",  # Bremen
            "West",  # Hamburg
            "West",  # Hessen (Hesse)
            "East",  # Mecklenburg-Vorpommern
            "West",  # Niedersachsen
            "West",  # Nordrhein-Westfalen
            "West",  # Rheinland-Pfalz
            "West",  # Saarland
            "East",  # Sachsen (Saxony)
            "East",  # Sachsen-Anhalt
            "West",  # Schleswig-Holstein
            "East",  # Thüringen (Thuringia)
            "West",  # Frankfurt (Hessen)
            "West",  # Munich (Bayern)
            "West",  # Aschaffenburg (Bayern)
        ]
    }
    
    return pd.DataFrame(states_data)

