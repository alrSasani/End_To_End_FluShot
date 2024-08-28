import pandas as pd
import json

def process_data_from_csv(file):
    df = pd.read_csv(file)
    data = df.to_dict(orient='records')[0]
    return data,df

def process_data_from_json(file):
    data = json.load(file)
    return data

def create_dataframe_from_input(data):
    input_data = {col: [data.get(col)] for col in COLUMN_NAMES}
    df = pd.DataFrame(input_data)
    return df

COLUMN_NAMES = [
       'h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds',
       'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands',
       'behavioral_large_gatherings', 'behavioral_outside_home',
       'behavioral_touch_face', 'doctor_recc_h1n1', 'doctor_recc_seasonal',
       'chronic_med_condition', 'child_under_6_months', 'health_worker',
       'health_insurance', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
       'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
       'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group',
       'education', 'race', 'sex', 'income_poverty', 'marital_status',
       'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa',
       'household_adults', 'household_children', 'employment_industry',
       'employment_occupation'
       ]

