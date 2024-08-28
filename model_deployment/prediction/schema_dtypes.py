import numpy as np
H1N1_CONCERN_CHOICES = [
    (2.0, 'Moderately concerned'),
    (1.0, 'Not very concerned'),
    (3.0, 'Very concerned'),
    (0.0, 'Not concerned at all'),
    (np.nan,np.nan),
]

H1N1_KNOWLEDGE_CHOICES = [
    (1.0, 'Some knowledge'),
    (2.0, 'Good knowledge'),
    (0.0, 'No knowledge'),
    (np.nan,np.nan),
]

BEHAVIORAL_CHOICES = [
    (0.0, 'No'),
    (1.0, 'Yes'),
    (np.nan,np.nan),
]

OPINION_CHOICES = [
    (1.0, 'Very Low'),
    (2.0, 'Low'),
    (3.0, 'Medium'),
    (4.0, 'High'),
    (5.0, 'Very High'),
    (np.nan,np.nan),
]

AGE_GROUP_CHOICES = [
    ('65+ Years', '65+ Years'),
    ('55 - 64 Years', '55 - 64 Years'),
    ('45 - 54 Years', '45 - 54 Years'),
    ('18 - 34 Years', '18 - 34 Years'),
    ('35 - 44 Years', '35 - 44 Years'),
    (np.nan,np.nan),
]

EDUCATION_CHOICES = [
    ('College Graduate', 'College Graduate'),
    ('Some College', 'Some College'),
    ('12 Years', '12 Years'),
    ('< 12 Years', '< 12 Years'),
    (np.nan,np.nan),
]

RACE_CHOICES = [
    ('White', 'White'),
    ('Black', 'Black'),
    ('Hispanic', 'Hispanic'),
    ('Other or Multiple', 'Other or Multiple'),
    (np.nan,np.nan),
]

SEX_CHOICES = [
    ('Female', 'Female'),
    ('Male', 'Male'),
    (np.nan,np.nan),
]

INCOME_CHOICES = [
    ('<= $75,000, Above Poverty', '<= $75,000, Above Poverty'),
    ('> $75,000', '> $75,000'),
    ('Below Poverty', 'Below Poverty'),
    (np.nan,np.nan),
]

MARITAL_STATUS_CHOICES = [
    ('Married', 'Married'),
    ('Not Married', 'Not Married'),
    (np.nan,np.nan),
]

RENT_OR_OWN_CHOICES = [
    ('Own', 'Own'),
    ('Rent', 'Rent'),
    (np.nan,np.nan),
]

EMPLOYMENT_STATUS_CHOICES = [
    ('Employed', 'Employed'),
    ('Not in Labor Force', 'Not in Labor Force'),
    ('Unemployed', 'Unemployed'),
    (np.nan,np.nan),
]

HHS_GEO_REGION_CHOICES = [
    ('lzgpxyit', 'lzgpxyit'),
    ('fpwskwrf', 'fpwskwrf'),
    ('qufhixun', 'qufhixun'),
    ('oxchjgsf', 'oxchjgsf'),
    ('kbazzjca', 'kbazzjca'),
    ('bhuqouqj', 'bhuqouqj'),
    ('mlyzmhmf', 'mlyzmhmf'),
    ('lrircsnp', 'lrircsnp'),
    ('atmpeygn', 'atmpeygn'),
    ('dqpwygqj', 'dqpwygqj'),
    (np.nan,np.nan),
]

CENSUS_MSA_CHOICES = [
    ('MSA, Not Principle  City', 'MSA, Not Principle  City'),
    ('MSA, Principle City', 'MSA, Principle City'),
    ('Non-MSA', 'Non-MSA'),
    (np.nan,np.nan),
]

HOUSEHOLD_ADULTS_CHOICES = [
    (1.0, '1'),
    (0.0, '0'),
    (2.0, '2'),
    (3.0, '3+'),
    (np.nan,np.nan),
]

HOUSEHOLD_CHILDREN_CHOICES = [
    (0.0, '0'),
    (1.0, '1'),
    (2.0, '2'),
    (3.0, '3+'),
    (np.nan,np.nan),
]

EMPLOYMENT_INDUSTRY_CHOICES = [
 ('fcxhlnwr', 'fcxhlnwr'),
 ('wxleyezf', 'wxleyezf'),
 ('ldnlellj', 'ldnlellj'),
 ('pxcmvdjn', 'pxcmvdjn'),
 ('atmlpfrs', 'atmlpfrs'),
 ('arjwrbjb', 'arjwrbjb'),
 ('xicduogh', 'xicduogh'),
 ('mfikgejo', 'mfikgejo'),
 ('vjjrobsf', 'vjjrobsf'),
 ('rucpziij', 'rucpziij'),
 ('xqicxuve', 'xqicxuve'),
 ('saaquncn', 'saaquncn'),
 ('cfqqtusy', 'cfqqtusy'),
 ('nduyfdeo', 'nduyfdeo'),
 ('mcubkhph', 'mcubkhph'),
 ('wlfvacwt', 'wlfvacwt'),
 ('dotnnunm', 'dotnnunm'),
 ('haxffmxo', 'haxffmxo'),
 ('msuufmds', 'msuufmds'),
 ('phxvnwax', 'phxvnwax'),
 ('qnlwzans', 'qnlwzans'),
 (np.nan,np.nan),
 ]

EMPLOYMENT_OCCUPATION_CHOICES = [
 ('xtkaffoo', 'xtkaffoo'),
 ('mxkfnird', 'mxkfnird'),
 ('emcorrxb', 'emcorrxb'),
 ('cmhcxjea', 'cmhcxjea'),
 ('xgwztkwe', 'xgwztkwe'),
 ('hfxkjkmi', 'hfxkjkmi'),
 ('qxajmpny', 'qxajmpny'),
 ('xqwwgdyp', 'xqwwgdyp'),
 ('kldqjyjy', 'kldqjyjy'),
 ('uqqtjvyb', 'uqqtjvyb'),
 ('tfqavkke', 'tfqavkke'),
 ('ukymxvdu', 'ukymxvdu'),
 ('vlluhbov', 'vlluhbov'),
 ('oijqvulv', 'oijqvulv'),
 ('ccgxvspp', 'ccgxvspp'),
 ('bxpfxfdn', 'bxpfxfdn'),
 ('haliazsg', 'haliazsg'),
 ('rcertsgn', 'rcertsgn'),
 ('xzmlyyjv', 'xzmlyyjv'),
 ('dlvbwzss', 'dlvbwzss'),
 ('hodpvpew', 'hodpvpew'),
 ('dcjcmpih', 'dcjcmpih'),
 ('pvmttkik', 'pvmttkik'),
 (np.nan,np.nan),
 ]
