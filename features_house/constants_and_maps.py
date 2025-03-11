import numpy as np


def language_mapping_dict():
    _eng_level_mapping = {
            # A1-A2 category
            'a1-a2': 1,
            'A1 - Beginner': 1,
            "A2 - Beginner": 1,
            "Beginner": 1,
            "A2": 1,
            'A1-A2': 1,
            'a2': 1,
            'A2 - Pre-intermediate': 1,
            ' A1 - Beginner': 1,
            
            # B1-B2 category
            'b1-b2': 2,
            'B1 - Intermediate': 2,
            'B1-B2': 2,
            'B2 - Upper-intermediate': 2,
            'Intermediate': 2,
            
            # C1-C2 category
            'c1-c2': 3,
            'C1 - Advanced': 3,
            'C2 - Proficient': 3,
            "C1": 3,
            'C1-C2': 3,
            'Advanced': 3,
            "Native": 3,
            'Fluent': 3,
            
            # Not sure category
            "I'm not sure": -1,
            "unknown": -1,
            "Muttersprache": 3,
            'Mother Tongue': 3,
            'mother tongue': 3,
            'Mother tongue': 3,
            "0": -1,
            "English": -1,
            np.nan: -1,
    }
    
    return _eng_level_mapping


def education_mapping():
    _education_mapping = {
            # No Formal Education
            None: "No Formal Education",
            "None": "No Formal Education",
            
            # Secondary Education
            "High school diploma": "Secondary Education",
            "Secondary school": "Secondary Education",
            "High school": "Secondary Education",
            "Mittlere Reife": "Secondary Education",
            "MSA": "Secondary Education",
            "Realschule": "Secondary Education",
            "Mittelschule": "Secondary Education",
            "mittlere reife": "Secondary Education",
            "Mittelschulabschluss": "Secondary Education",
            "Quali": "Secondary Education",
            "A levels": "Secondary Education",
            "Förderschule": "Secondary Education",
            
            # Vocational Training
            "Ausbildung / Vocational training": "Vocational Training",
            "Ausbildung/Vocational training": "Vocational Training",
            "Techniker": "Vocational Training",
            "Meister": "Vocational Training",
            "Fachwirt": "Vocational Training",
            "Staatlich geprüfter Techniker": "Vocational Training",
            
            # Lower-Tier Diplomas
            "Hauptschule": "Secondary Education",
            "Haupt": "Secondary Education",
            "hauptschule": "Secondary Education",
            "Hauptschulabschluss": "Secondary Education",
            "Hauptschulabschluß": "Secondary Education",
            "Qualifizierter Hauptschulabschluss": "Secondary Education",
            "Erweiterter Hauptschulabschluss": "Secondary Education",
            "HA10": "Secondary Education",
            
            # Undergraduate Degree
            "Bachelor degree": "Undergraduate Degree",
            "Bachelor's degree": "Undergraduate Degree",
            "Bachelor": "Undergraduate Degree",
            "University": "Undergraduate Degree",
            "Undergraduate": "Undergraduate Degree",
            "Hochschule": "Undergraduate Degree",
            "Diploma": "Undergraduate Degree",
            "Diplom (FH)": "Undergraduate Degree",
            "Associate degree": "Undergraduate Degree",
            
            # Graduate Degree
            "Master degree": "Graduate Degree",
            "Master's degree": "Graduate Degree",
            "Master of Business Administration": "Graduate Degree",
            "Diplom": "Graduate Degree",
            "Diplom Ingenieur": "Graduate Degree",
            "Fachhochschulreife": "Graduate Degree",
            "Fachabitur": "Graduate Degree",
            "Staatsexamen": "Graduate Degree",
            
            # Doctorate
            "Doctorate": "Doctorate",
            
            # Other (Unclear / Edge Cases)
            "Driver": "Other"
        
    }
    
    return _education_mapping


def location_mapping():
    # Creating a dictionary to map the user-provided locations to the standardized locations used in the DataFrame
    _location_mapping = {
            'Niedersachsen (Lower Saxony)': 'Niedersachsen',
            'aschaffenburg': 'Bayern (Bavaria)',
            'Saxony anhalt': 'Sachsen-Anhalt',
            'Saxony': 'Sachsen (Saxony)',
            'North Rhine-Westphalia': 'Nordrhein-Westfalen',
            'frankfurt': 'Hessen (Hesse)',
            'Munich': 'Bayern (Bavaria)',
            'Hessen (Hesse)': 'Hessen (Hesse)',
            'kampala': "other",  # Not a German state
            'Ghana': "other",  # Not a German state
            'Bavaria': 'Bayern (Bavaria)',
            'Saarland': 'Saarland',
            'Thüringen': 'Thüringen (Thuringia)',
            'Sachsen-Anhalt': 'Sachsen-Anhalt',
            'Bayern': 'Bayern (Bavaria)',
            'Baden-Württemberg': 'Baden-Württemberg',
            'Hessen': 'Hessen (Hesse)',
            'Schleswig-Holstein': 'Schleswig-Holstein',
            'Rheinland-Pfalz': 'Rheinland-Pfalz',
            'Bremen': 'Bremen',
            'Mecklenburg-Vorpommern': 'Mecklenburg-Vorpommern',
            'Hamburg': 'Hamburg',
            'Nordrhein-Westfalen': 'Nordrhein-Westfalen',
            'Niedersachsen': 'Niedersachsen',
            'Sachsen': 'Sachsen (Saxony)',
            'Brandenburg': 'Brandenburg',
            'Berlin': 'Berlin',
            'Nordrhein-Westfalen (North Rhine-Westphalia)': 'Nordrhein-Westfalen',
            'Bayern (Bavaria)': 'Bayern (Bavaria)',
            'Sachsen (Saxony)': 'Sachsen (Saxony)',
            "not specified": "not specified"
    }
    return _location_mapping


def work_experience_mapping():
    work_experience_mapping = {
            # Valid responses to "How long have you worked in Germany?"
            "More than 12 months": "More than 12 months",
            "Mehr als 12 Monate": "More than 12 months",  # German equivalent
            "I haven't worked in Germany": "I haven't worked in Germany",
            "Ich habe noch nie in Deutschland gearbeitet": "I haven't worked in Germany",  # German equivalent
            "Less than 6 months": "Less than 6 months",
            "Weniger als 6 Monate": "Less than 6 months",  # German equivalent
            "6-12 months": "6-12 months",
            "6 bis 12 Monate": "6-12 months",  # German equivalent
            
            # Responses that do not match the question → Categorized as "Other"
            "not specified": "not specified",
            "Abitur / Realschule": "Other",
            "Ausbildung": "Other",
            "Keinen": "Other",
            "B.A/B.Sc": "Other",
            "M.A/M.Sc": "Other",
            "Hauptschule": "Other",
            "Hauptschulabschluss": "Other",
            "PhD/MD": "Other",
            "Hauptschulabschluss + Musik Studium (Digitale-Musik Produktion, Schwehrpunkt Filmmusik) auf privater Ebene": "Other",
            "NO": "Other",
            "Yes": "Other",
            "Qualifizierte Hauptschulabschluss": "Other",
            "Hauptschulabschluss und Training als Pflegeassistenz": "Other",
            "Erweiterter Hauptschulabschluss": "Other",
            "leider nur hauptschul-abschluss": "Other",
            "QA": "Other",
            "Master im Betriebswirtschaftslehre": "Other",
            "Fachhochschulreife+Bankausbildung": "Other",
            
    }
    
    return work_experience_mapping


def states_to_continent_map():
    country_to_continent = {
            "KE": "Africa",  # Kenya
            "ZA": "Africa",  # South Africa
            "unknown": "Unknown",  # Not specified
            "IL": "Asia",  # Israel
            "CA": "North America",  # Canada
            "QA": "Asia",  # Qatar
            "NG": "Africa",  # Nigeria
            "BH": "Asia",  # Bahrain
            "ZW": "Africa",  # Zimbabwe
            "BD": "Asia",  # Bangladesh
            "UA": "Europe",  # Ukraine
            "PE": "South America",  # Peru
            "CR": "North America",  # Costa Rica (Central America is part of North America)
            "RU": "Europe",  # Russia (commonly assigned to Europe in many schemes)
            "US": "North America",  # United States
            "YE": "Asia",  # Yemen
            "DE": "Europe",  # Germany
            "CO": "South America",  # Colombia
            "PK": "Asia",  # Pakistan
            "LC": "North America",  # Saint Lucia (Caribbean)
            "DK": "Europe",  # Denmark
            "SI": "Europe",  # Slovenia
            "SA": "Asia",  # Saudi Arabia
            "GM": "Africa",  # The Gambia
            "GH": "Africa",  # Ghana
            "IN": "Asia",  # India
            "MA": "Africa",  # Morocco
            "CH": "Europe",  # Switzerland
            "PT": "Europe",  # Portugal
            "MY": "Asia",  # Malaysia
            "CL": "South America",  # Chile
            "AF": "Asia",  # Afghanistan
            "NE": "Africa",  # Niger
            "IT": "Europe",  # Italy
            "IR": "Asia",  # Iran
            "AT": "Europe",  # Austria
            "CN": "Asia",  # China
            "ES": "Europe",  # Spain
            "PH": "Asia",  # Philippines
            "TN": "Africa",  # Tunisia
            "EG": "Africa",  # Egypt
            "AL": "Europe",  # Albania
            "CD": "Africa",  # Democratic Republic of the Congo
            "SY": "Asia",  # Syria
            "GT": "North America",  # Guatemala (Central America is part of North America)
            "AE": "Asia",  # United Arab Emirates
            "VN": "Asia",  # Vietnam
            "BO": "South America",  # Bolivia
            "SG": "Asia",  # Singapore
            "LU": "Europe",  # Luxembourg
            "GR": "Europe",  # Greece
            "ET": "Africa",  # Ethiopia
            "CZ": "Europe",  # Czechia
            "KW": "Asia",  # Kuwait
            "NL": "Europe",  # Netherlands
            "AO": "Africa",  # Angola
            "TR": "Asia",  # Turkey (mostly in Asia by common classification)
            "KG": "Asia",  # Kyrgyzstan
            "PL": "Europe",  # Poland
            "LY": "Africa",  # Libya
            "CM": "Africa",  # Cameroon
            "BR": "South America",  # Brazil
            "ZM": "Africa",  # Zambia
            "MK": "Europe",  # North Macedonia
            "NZ": "Oceania",  # New Zealand
            "LT": "Europe",  # Lithuania
            "BG": "Europe",  # Bulgaria
            "OM": "Asia",  # Oman
            "RO": "Europe",  # Romania
            "GE": "Asia",  # Georgia (located in the Caucasus, often classified as Asia)
            "BA": "Europe",  # Bosnia and Herzegovina
            "BE": "Europe",  # Belgium
            "HR": "Europe",  # Croatia
            "LV": "Europe",  # Latvia
            "SL": "Africa",  # Sierra Leone
            "GB": "Europe",  # United Kingdom
            "IE": "Europe",  # Ireland
            "NP": "Asia",  # Nepal
            "AU": "Oceania",  # Australia
            "JP": "Asia",  # Japan
            "TT": "North America",  # Trinidad and Tobago (Caribbean)
            "TZ": "Africa",  # Tanzania
            "SE": "Europe",  # Sweden
            "FR": "Europe",  # France
            "EC": "South America",  # Ecuador
            "BW": "Africa",  # Botswana
            "PG": "Oceania",  # Papua New Guinea
            "RW": "Africa",  # Rwanda
            "PS": "Asia",  # Palestine
            "JO": "Asia",  # Jordan
            "MD": "Europe",  # Moldova
            "HK": "Asia",  # Hong Kong
            "KR": "Asia",  # South Korea
            "LB": "Asia",  # Lebanon
            "RS": "Europe",  # Serbia
            "UY": "South America",  # Uruguay
            "PA": "North America",  # Panama
            "SD": "Africa",  # Sudan
            "MX": "North America",  # Mexico
            "LR": "Africa",  # Liberia
            "FJ": "Oceania",  # Fiji
            "HU": "Europe",  # Hungary
            "PY": "South America",  # Paraguay
            "VE": "South America",  # Venezuela
            "UG": "Africa",  # Uganda
            "SO": "Africa",  # Somalia
            "SB": "Oceania",  # Solomon Islands
            "AZ": "Asia",  # Azerbaijan
            "ML": "Africa",  # Mali
            "IQ": "Asia",  # Iraq
            "ID": "Asia",  # Indonesia
            "LK": "Asia",  # Sri Lanka
            "DZ": "Africa",  # Algeria
            "AM": "Asia",  # Armenia
            "GN": "Africa",  # Guinea
            "TH": "Asia",  # Thailand
            "MV": "Asia",  # Maldives
            "SN": "Africa",  # Senegal
            "CI": "Africa",  # Ivory Coast
            "JM": "North America",  # Jamaica
            "BI": "Africa",  # Burundi
            "TW": "Asia",  # Taiwan
            "AR": "South America",  # Argentina
            "NO": "Europe",  # Norway
            "MM": "Asia",  # Myanmar
            "CY": "Asia",  # Cyprus (often classified as Asia geopolitically)
            "unable_to_identify": "Unknown",  # Not specified
            "BY": "Europe",  # Belarus
            "MU": "Africa",  # Mauritius
            "FI": "Europe",  # Finland
            "KZ": "Asia",  # Kazakhstan (often assigned to Asia despite partial European territory)
            "TG": "Africa",  # Togo
            "MN": "Asia",  # Mongolia
            "PR": "North America",  # Puerto Rico (Caribbean, part of North America)
            "CV": "Africa",  # Cape Verde
            "SK": "Europe",  # Slovakia
            "MG": "Africa",  # Madagascar
            "HN": "North America",  # Honduras
            "DJ": "Africa",  # Djibouti
            "BM": "North America",  # Bermuda (geographically in the North Atlantic, part of North America)
            "HT": "North America",  # Haiti
            "DO": "North America",  # Dominican Republic
            "MT": "Europe",  # Malta
            "IS": "Europe",  # Iceland
            "BJ": "Africa",  # Benin
            "MO": "Asia",  # Macau
            "TC": "North America",  # Turks and Caicos Islands (Caribbean)
            "GW": "Africa",  # Guinea-Bissau
            "CU": "North America",  # Cuba
            "MZ": "Africa",  # Mozambique
            "UZ": "Asia",  # Uzbekistan
            "BF": "Africa",  # Burkina Faso
            "EE": "Europe",  # Estonia
            "TJ": "Asia",  # Tajikistan
            "ER": "Africa",  # Eritrea
            "KH": "Asia",  # Cambodia
            "BN": "Asia",  # Brunei
            "GA": "Africa",  # Gabon
            "NI": "North America",  # Nicaragua
            "RE": "Africa",  # Réunion (French overseas department in Africa)
            "TD": "Africa",  # Chad
            "XK": "Europe",  # Kosovo (disputed, but usually grouped with Europe)
            "BS": "North America",  # Bahamas
            "TM": "Asia",  # Turkmenistan
            "SS": "Africa",  # South Sudan
            "SC": "Africa",  # Seychelles
            "MW": "Africa",  # Malawi
            "PF": "Oceania",  # French Polynesia
            "SR": "South America",  # Suriname
            "GL": "North America",  # Greenland (geographically part of North America)
            "KM": "Africa",  # Comoros
            "MC": "Europe",  # Monaco
            "AD": "Europe",  # Andorra
            "GQ": "Africa",  # Equatorial Guinea
            "BB": "North America",  # Barbados
            "LA": "Asia",  # Laos
            "MR": "Africa",  # Mauritania
            "AW": "North America",  # Aruba
            "BT": "Asia",  # Bhutan
            "GP": "North America",  # Guadeloupe (Caribbean)
            "SV": "North America",  # El Salvador
            "LI": "Europe",  # Liechtenstein
            "GI": "Europe",  # Gibraltar
            "AG": "North America",  # Antigua and Barbuda (Caribbean)
            "FO": "Europe",  # Faroe Islands
            "CW": "North America",  # Curaçao (Caribbean)
            "CF": "Africa",  # Central African Republic
            "GF": "South America",  # French Guiana (overseas region of France in South America)
            "NC": "Oceania",  # New Caledonia
            "GG": "Europe",  # Guernsey
            "VC": "North America",  # Saint Vincent and the Grenadines (Caribbean)
            "IM": "Europe",  # Isle of Man
            "MH": "Oceania",  # Marshall Islands
            "VA": "Europe",  # Vatican City
            "TO": "Oceania",  # Tonga
            "NU": "Oceania",  # Niue
            "BZ": "North America",  # Belize
            "LS": "Africa",  # Lesotho
            "JE": "Europe",  # Jersey
            "CG": "Africa",  # Republic of the Congo
            "YT": "Africa",  # Mayotte
            "AS": "Oceania",  # American Samoa
            "VU": "Oceania",  # Vanuatu
            "KI": "Oceania",  # Kiribati
            "ME": "Europe",  # Montenegro
    }
    return country_to_continent
