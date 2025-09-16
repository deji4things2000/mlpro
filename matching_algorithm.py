import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import logging
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Dictionaries and Constants ---
COUNTRY_ABBREVIATIONS = {
    "AF": "Afghanistan", "AO": "Angola", "AI": "Anguilla", "AX": "Aland Islands",
    "AL": "Albania", "AD": "Andorra", "AE": "United Arab Emirates", "AR": "Argentina", 
    "AM": "Armenia", "AS": "American Samoa", "AQ": "Antarctica", "TF": "French Southern Territories",
    "AG": "Antigua and Barbuda", "AU": "Australia", "AT": "Austria", "AZ": "Azerbaijan",
    "BI": "Burundi", "BE": "Belgium", "BJ": "Benin", "BQ": "Bonaire, Sint Eustatius and Saba",
    "BF": "Burkina Faso", "BD": "Bangladesh", "BG": "Bulgaria", "BH": "Bahrain", "BS": "Bahamas",
    "BA": "Bosnia and Herzegovina", "BL": "Saint Bartholomew", "BY": "Belarus", "BZ": "Belize",
    "BM": "Bermuda", "BO": "Bolivia, Plurinational State of", "BR": "Brazil", "BB": "Barbados",
    "BN": "Brunei Darussalam", "BT": "Bhutan", "BV": "Bouvet Island", "BW": "Botswana",
    "CF": "Central African Republic", "CA": "Canada", "CC": "Cocos (Keeling) Islands",
    "CH": "Switzerland", "CL": "Chile", "CN": "China", "CI": "Cote d'Ivoire", "CM": "Cameroon",
    "CD": "Congo, The Democratic Republic of the", "CG": "Congo", "CK": "Cook Islands",
    "CO": "Colombia", "KM": "Comoros", "CV": "Cabo Verde", "CR": "Costa Rica", "CU": "Cuba",
    "CW": "Curacao", "CX": "Christmas Island", "KY": "Cayman Islands", "CY": "Cyprus",
    "CZ": "Czechia", "DE": "Germany", "DJ": "Djibouti", "DM": "Dominica", "DK": "Denmark",
    "DO": "Dominican Republic", "DZ": "Algeria", "EC": "Ecuador", "EG": "Egypt", "ER": "Eritrea",
    "EH": "Western Sahara", "ES": "Spain", "EE": "Estonia", "ET": "Ethiopia", "FI": "Finland",
    "FJ": "Fiji", "FK": "Falkland Islands (Malvinas)", "FR": "France", "FO": "Faroe Islands",
    "FM": "Micronesia, Federated States of", "GA": "Gabon", "GB": "United Kingdom", "GE": "Georgia",
    "GG": "Guernsey", "GH": "Ghana", "GI": "Gibraltar", "GN": "Guinea", "GP": "Guadeloupe",
    "GM": "Gambia", "GW": "Guinea-Bissau", "GQ": "Equatorial Guinea", "GR": "Greece",
    "GD": "Grenada", "GL": "Greenland", "GT": "Guatemala", "GF": "French Guiana", "GU": "Guam", 
    "GY": "Guyana", "HK": "Hong Kong", "HM": "Heard Island and McDonald Islands", "HN": "Honduras", 
    "HR": "Croatia", "HT": "Haiti", "HU": "Hungary", "ID": "Indonesia", "IM": "Isle of Man", 
    "IN": "India", "IO": "British Indian Ocean Territory", "IE": "Ireland", "IR": "Iran, Islamic Republic of",
    "IQ": "Iraq", "IS": "Iceland", "IL": "Israel", "IT": "Italy", "JM": "Jamaica", "JE": "Jersey",
    "JO": "Jordan", "JP": "Japan", "KZ": "Kazakhstan", "KE": "Kenya", "KG": "Kyrgyzstan",
    "KH": "Cambodia", "KI": "Kiribati", "KN": "Saint Kitts and Nevis", "KR": "Korea, Republic of",
    "KW": "Kuwait", "LA": "Lao People's Democratic Republic", "LB": "Lebanon", "LR": "Liberia",
    "LY": "Libya", "LC": "Saint Lucia", "LI": "Liechtenstein", "LK": "Sri Lanka", "LS": "Lesotho",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MO": "Macao",
    "MF": "Saint Martin (French part)", "MA": "Morocco", "MC": "Monaco",
    "MD": "Moldova, Republic of", "MG": "Madagascar", "MV": "Maldives", "MX": "Mexico",
    "MH": "Marshall Islands", "MK": "North Macedonia", "ML": "Mali", "MT": "Malta", "MM": "Myanmar",
    "ME": "Montenegro", "MN": "Mongolia", "MP": "Northern Mariana Islands", "MZ": "Mozambique",
    "MR": "Mauritania", "MS": "Montserrat", "MQ": "Martinique", "MU": "Mauritius", "MW": "Malawi",
    "MY": "Malaysia", "YT": "Mayotte", "NA": "Namibia", "NC": "New Caledonia", "NE": "Niger",
    "NF": "Norfolk Island", "NG": "Nigeria", "NI": "Nicaragua", "NU": "Niue", "NL": "Netherlands",
    "NO": "Norway", "NP": "Nepal", "NR": "Nauru", "NZ": "New Zealand", "OM": "Oman", "PK": "Pakistan",
    "PA": "Panama", "PE": "Peru", "PH": "Philippines", "PW": "Palau",
    "PG": "Papua New Guinea", "PL": "Poland", "PR": "Puerto Rico",
    "KP": "Korea, Democratic People's Republic of", "PT": "Portugal", "PY": "Paraguay",
    "PS": "Palestine, State of", "PF": "French Polynesia", "QA": "Qatar", "RE": "Reunion",
    "RO": "Romania", "RU": "Russian Federation", "RW": "Rwanda", "SA": "Saudi Arabia", "SD": "Sudan",
    "SN": "Senegal", "SG": "Singapore", "GS": "South Georgia and the South Sandwich Islands",
    "SH": "Saint Helena, Ascension and Tristan da Cunha", "SJ": "Svalbard and Jan Mayen",
    "SB": "Solomon Islands", "SL": "Sierra Leone", "SV": "El Salvador", "SM": "San Marino",
    "SO": "Somalia", "PM": "Saint Pierre and Miquelon", "RS": "Serbia", "SS": "South Sudan",
    "ST": "Sao Tome and Principe", "SR": "Suriname", "SK": "Slovakia", "SI": "Slovenia",
    "SE": "Sweden", "SZ": "Eswatini", "SX": "Sint Maarten (Dutch part)", "SC": "Seychelles",
    "SY": "Syrian Arab Republic", "TC": "Turks and Caicos Islands", "TD": "Chad", "TG": "Togo",
    "TH": "Thailand", "TJ": "Tajikistan", "TK": "Tokelau", "TM": "Turkmenistan",
    "TL": "Timor-Leste", "TO": "Tonga", "TT": "Trinidad and Tobago", "TN": "Tunisia",
    "TR": "Turkey", "TV": "Tuvalu", "TW": "Taiwan, Province of China",
    "TZ": "Tanzania, United Republic of", "UG": "Uganda", "UA": "Ukraine",
    "UM": "United States Minor Outlying Islands", "UY": "Uruguay", "US": "United States",
    "UZ": "Uzbekistan", "VA": "Holy See (Vatican City State)",
    "VC": "Saint Vincent and the Grenadines", "VE": "Venezuela", "VG": "Virgin Islands, British",
    "VI": "Virgin Islands, U.S.", "VN": "Viet Nam", "VU": "Vanuatu", "WF": "Wallis and Futuna",
    "WS": "Samoa", "YE": "Yemen", "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe"
}

def simple_preprocess(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text_lower = text.lower()
    text_alnum = ''.join([char for char in text_lower if char.isalnum() or char.isspace()])
    text_normalized = ' '.join(text_alnum.split())
    corporate_terms = ['inc', 'llc', 'ltd', 'co', 'corp', 'corporation', 'limited', 
                       'company', 'pte', 'llp', 'plc', 'sa', 'gmbh', 'group', 'holding', 'enterprise']
    words = [word for word in text_normalized.split() if word not in corporate_terms]
    final_text = ' '.join(words)
    return final_text if final_text.strip() else text_normalized

def standardize_country(country_name):
    if pd.isna(country_name) or not isinstance(country_name, str):
        return ""
    country_name_upper = country_name.strip().upper()
    if country_name_upper in COUNTRY_ABBREVIATIONS:
        return COUNTRY_ABBREVIATIONS[country_name_upper]
    processed_name = simple_preprocess(country_name)
    for k, v in COUNTRY_ABBREVIATIONS.items():
        if v.lower() == processed_name:
            return v
    return country_name

def detect_fields(df, primary_keywords, secondary_keywords):
    name_field, country_field = None, None
    for col in df.columns:
        col_lower = col.lower()
        if country_field is None and any(k in col_lower for k in secondary_keywords):
            country_field = col
        if name_field is None and any(k in col_lower for k in primary_keywords):
            name_field = col
    
    if name_field is None:
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and col != country_field:
                non_null_count = df[col].count()
                if non_null_count > 0 and (df[col].nunique() / non_null_count) > 0.5:
                    name_field = col
                    logging.warning(f"Auto-detected name field '{name_field}' based on high uniqueness.")
                    break
    
    if name_field is None and not df.empty:
        name_field = df.columns[0]
        logging.warning(f"Could not detect a name field. Falling back to the first column: '{name_field}'.")
    
    return name_field, country_field

def combined_similarity(row1, row2, fields1, fields2, weights):
    name_field1, country_field1 = fields1
    name_field2, country_field2 = fields2
    name_weight, country_weight = weights
    
    score_name = fuzz.token_set_ratio(row1[name_field1], row2[name_field2])
    total_score = score_name * name_weight
    total_weight = name_weight
    
    if (country_field1 and country_field2 and 
        pd.notna(row1[country_field1]) and pd.notna(row2[country_field2]) and 
        row1[country_field1] and row2[country_field2]):
        score_country = fuzz.token_set_ratio(row1[country_field1], row2[country_field2])
        total_score += score_country * country_weight
        total_weight += country_weight
    
    return total_score / total_weight if total_weight > 0 else 0

def match_suppliers_optimized(file1_path, file2_path, output_path, threshold=83, 
                             name_weight=1, country_weight=1, n_neighbors=90, 
                             file1_cols=None, file2_cols=None):
    start_time = time.time()
    
    # Load data
    logging.info(f"Loading data from {file1_path} and {file2_path}...")
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # Field Detection (with Manual Override)
    if file1_cols:
        name_field1 = file1_cols.get('name')
        country_field1 = file1_cols.get('country')
        logging.info("Using manually specified columns for File 1.")
    else:
        logging.info("Detecting name and country fields for File 1...")
        name_keywords = ['name', 'company', 'supplier', 'vendor']
        country_keywords = ['country', 'location']
        name_field1, country_field1 = detect_fields(df1, name_keywords, country_keywords)
    
    if file2_cols:
        name_field2 = file2_cols.get('name')
        country_field2 = file2_cols.get('country')
        logging.info("Using manually specified columns for File 2.")
    else:
        logging.info("Detecting name and country fields for File 2...")
        name_keywords = ['name', 'company', 'supplier', 'vendor']
        country_keywords = ['country', 'location']
        name_field2, country_field2 = detect_fields(df2, name_keywords, country_keywords)
    
    logging.info(f"File 1 using: Name='{name_field1}', Country='{country_field1}'")
    logging.info(f"File 2 using: Name='{name_field2}', Country='{country_field2}'")
    
    if not name_field1 or not name_field2:
        logging.error("Could not detect name fields in both files. Aborting.")
        return None
    
    # Preprocessing
    logging.info("Preprocessing data...")
    df1['processed_name'] = df1[name_field1].apply(simple_preprocess)
    df2['processed_name'] = df2[name_field2].apply(simple_preprocess)
    
    df1['processed_country'] = df1[country_field1].apply(standardize_country).apply(simple_preprocess) if country_field1 else ""
    df2['processed_country'] = df2[country_field2].apply(standardize_country).apply(simple_preprocess) if country_field2 else ""
    
    df1['combined_text'] = df1['processed_name'] + " " + df1['processed_country']
    df2['combined_text'] = df2['processed_name'] + " " + df2['processed_country']
    
    # Vectorization and Nearest Neighbor Search
    logging.info("Vectorizing text and finding candidates...")
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 2))
    
    if df2.empty:
        logging.error("File 2 is empty. Cannot perform matching.")
        return None
    
    tfidf_matrix2 = vectorizer.fit_transform(df2['combined_text'])
    actual_n_neighbors = min(n_neighbors, len(df2))
    nn_model = NearestNeighbors(n_neighbors=actual_n_neighbors, metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix2)
    
    tfidf_matrix1 = vectorizer.transform(df1['combined_text'])
    distances, indices = nn_model.kneighbors(tfidf_matrix1)
    
    # Detailed Fuzzy Matching on Candidates
    logging.info("Performing detailed fuzzy matching on candidate pairs...")
    results = []
    matched_indices_df2 = set()
    
    original_cols1 = [col for col in df1.columns if '_processed' not in col and 'combined_text' not in col]
    original_cols2 = [col for col in df2.columns if '_processed' not in col and 'combined_text' not in col]
    
    for i in range(len(df1)):
        row1 = df1.iloc[i]
        best_score, best_match_idx = -1, -1
        
        for candidate_idx in indices[i]:
            if candidate_idx in matched_indices_df2:
                continue
            row2 = df2.iloc[candidate_idx]
            score = combined_similarity(
                row1, row2, 
                ('processed_name', 'processed_country'), 
                ('processed_name', 'processed_country'), 
                (name_weight, country_weight)
            )
            if score > best_score:
                best_score, best_match_idx = score, candidate_idx
        
        if best_score >= threshold and best_match_idx != -1:
            matched_indices_df2.add(best_match_idx)
            row_matched = df2.iloc[best_match_idx]
            result_row = {'Match_Score': best_score, 'Status': 'Matched'}
            result_row.update({f'File1_{col}': row1[col] for col in original_cols1})
            result_row.update({f'File2_{col}': row_matched[col] for col in original_cols2})
            results.append(result_row)
        else:
            result_row = {'Match_Score': best_score, 'Status': 'No Match (Below Threshold)'}
            result_row.update({f'File1_{col}': row1[col] for col in original_cols1})
            result_row.update({f'File2_{col}': np.nan for col in original_cols2})
            results.append(result_row)
    
    # Finalizing Results
    logging.info("Identifying unmatched records from File 2...")
    unmatched_df2_indices = set(df2.index) - matched_indices_df2
    for idx in unmatched_df2_indices:
        row_unmatched = df2.iloc[idx]
        result_row = {'Match_Score': 0, 'Status': 'Unmatched in File2'}
        result_row.update({f'File1_{col}': np.nan for col in original_cols1})
        result_row.update({f'File2_{col}': row_unmatched[col] for col in original_cols2})
        results.append(result_row)
    
    results_df = pd.DataFrame(results).sort_values('Match_Score', ascending=False).reset_index(drop=True)
    
    # Save results
    logging.info(f"Saving results to {output_path}...")
    if output_path.endswith('.csv'):
        results_df.to_csv(output_path, index=False)
    else:
        results_df.to_excel(output_path, index=False)
    
    # Reporting
    total_matches = len(results_df[results_df['Status'] == 'Matched'])
    min_len = min(len(df1), len(df2)) if len(df2) > 0 else len(df1)
    match_rate = (total_matches / min_len * 100) if min_len > 0 else 0
    
    logging.info("\n--- Matching Complete ---")
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
    logging.info(f"Total matches found: {total_matches}")
    logging.info(f"Match rate: {match_rate:.1f}%")
    logging.info(f"Results saved to {output_path}")
    
    return results_df

# --- Main Execution Block ---
if __name__ == "__main__":
    # Configuration
    file1_path = r"C:\Users\sa66389\bamp2p.csv"
    file2_path = r"C:\Users\sa66389\aravo_rel.csv"
    output_path = r"C:\Users\sa66389\aml_map.csv"
    
    # Manually specify column names to override detection
    file1_columns = {
        'name': 'SUPPLIER NAME',
        'country': 'COUNTRY'
    }
    
    file2_columns = {
        'name': 'Parent Name',
        'country': 'Parent Country HQ'  # IMPORTANT: Replace this with the real column name from aravo_rel.csv
    }
    
    try:
        results = match_suppliers_optimized(
            file1_path=file1_path,
            file2_path=file2_path,
            output_path=output_path,
            threshold=90,
            name_weight=3,
            country_weight=1,
            n_neighbors=80,
            file1_cols=file1_columns,
            file2_cols=file2_columns
        )
    except FileNotFoundError:
        logging.error("FATAL: Input file not found. Please check paths.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)