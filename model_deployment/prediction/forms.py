from django import forms
from .schema_dtypes import *

class UserDataForm(forms.Form):
    h1n1_concern = forms.ChoiceField(choices=H1N1_CONCERN_CHOICES, label='H1N1 Concern', required=False)
    h1n1_knowledge = forms.ChoiceField(choices=H1N1_KNOWLEDGE_CHOICES, label='H1N1 Knowledge', required=False)
    
    behavioral_antiviral_meds = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Behavioral: Antiviral Meds', required=False)
    behavioral_avoidance = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Behavioral: Avoidance', required=False)
    behavioral_face_mask = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Behavioral: Face Mask', required=False)
    behavioral_wash_hands = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Behavioral: Wash Hands', required=False)
    behavioral_large_gatherings = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Behavioral: Large Gatherings', required=False)
    behavioral_outside_home = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Behavioral: Outside Home', required=False)
    behavioral_touch_face = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Behavioral: Touch Face', required=False)
    doctor_recc_h1n1 = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Doctor Recommend H1N1', required=False)
    doctor_recc_seasonal = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Doctor Recommend Seasonal', required=False)
    chronic_med_condition = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Chronic Medical Condition', required=False)
    child_under_6_months = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Child Under 6 Months', required=False)
    health_worker = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Health Worker', required=False)
    health_insurance = forms.ChoiceField(choices=BEHAVIORAL_CHOICES, label='Health Insurance', required=False)
    
    opinion_h1n1_vacc_effective = forms.ChoiceField(choices=OPINION_CHOICES, label='Opinion: H1N1 Vaccine Effective', required=False)
    opinion_h1n1_risk = forms.ChoiceField(choices=OPINION_CHOICES, label='Opinion: H1N1 Risk', required=False)
    opinion_h1n1_sick_from_vacc = forms.ChoiceField(choices=OPINION_CHOICES, label='Opinion: H1N1 Sick from Vaccine', required=False)
    opinion_seas_vacc_effective = forms.ChoiceField(choices=OPINION_CHOICES, label='Opinion: Seasonal Vaccine Effective', required=False)
    opinion_seas_risk = forms.ChoiceField(choices=OPINION_CHOICES, label='Opinion: Seasonal Risk', required=False)
    opinion_seas_sick_from_vacc = forms.ChoiceField(choices=OPINION_CHOICES, label='Opinion: Seasonal Sick from Vaccine', required=False)
    
    age_group = forms.ChoiceField(choices=AGE_GROUP_CHOICES, label='Age Group', required=False)
    education = forms.ChoiceField(choices=EDUCATION_CHOICES, label='Education', required=False)
    race = forms.ChoiceField(choices=RACE_CHOICES, label='Race', required=False)
    sex = forms.ChoiceField(choices=SEX_CHOICES, label='Sex', required=False)
    income_poverty = forms.ChoiceField(choices=INCOME_CHOICES, label='Income Poverty', required=False)
    marital_status = forms.ChoiceField(choices=MARITAL_STATUS_CHOICES, label='Marital Status', required=False)
    rent_or_own = forms.ChoiceField(choices=RENT_OR_OWN_CHOICES, label='Rent or Own', required=False)
    employment_status = forms.ChoiceField(choices=EMPLOYMENT_STATUS_CHOICES, label='Employment Status', required=False)
    hhs_geo_region = forms.ChoiceField(choices=HHS_GEO_REGION_CHOICES, label='HHS Geo Region', required=False)
    census_msa = forms.ChoiceField(choices=CENSUS_MSA_CHOICES, label='Census MSA', required=False)
    household_adults = forms.ChoiceField(choices=HOUSEHOLD_ADULTS_CHOICES, label='Household Adults', required=False)
    household_children = forms.ChoiceField(choices=HOUSEHOLD_CHILDREN_CHOICES, label='Household Children', required=False)
    
    employment_industry = forms.ChoiceField(choices=EMPLOYMENT_INDUSTRY_CHOICES, label='Employment Industry', required=False)
    employment_occupation = forms.ChoiceField(choices=EMPLOYMENT_OCCUPATION_CHOICES, label='Employment Occupation', required=False)
