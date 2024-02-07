import pandas as pd
from google.cloud import bigquery
import os
import matplotlib.pyplot as plt 
import numpy as np
import statsmodels.api as sm

plt.rcParams.update({'font.size': 16})


#connection to Nero google cloud project to access STARR OMOP data is documented in these youtube tutorials 
#https://www.youtube.com/channel/UC6iGiAO1dKwuC2wOrxnKiNw
user_id = 'X'
nero_gcp_project = 'X'
cdm_project_id = 'X'
cdm_dataset_id = 'X'
work_project_id = nero_gcp_project
work_dataset_id = 'X'


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'X'
os.environ['GCLOUD_PROJECT'] = nero_gcp_project 
client = bigquery.Client(project=work_project_id)


##read in the kidney clinic visits dataframe
query="""
SELECT 
*
FROM`{work_project_id}.{work_dataset_id}.X` 
""".format_map({'cdm_project_id': cdm_project_id,
                'cdm_dataset_id': cdm_dataset_id, 
               'work_project_id': work_project_id,
                  'work_dataset_id': work_dataset_id})
kidney_visit_df = client.query(query).to_dataframe()
kidney_visit_df.head()


#read in the kidney clinic referral data frame
query="""
SELECT 
*
FROM`{work_project_id}.{work_dataset_id}.X` 
""".format_map({'cdm_project_id': cdm_project_id,
                'cdm_dataset_id': cdm_dataset_id, 
               'work_project_id': work_project_id,
                  'work_dataset_id': work_dataset_id})
kidney_referral_df = client.query(query).to_dataframe()
kidney_referral_df.head()


##assigns binary variables for each race according to the electronic health record values
race_values = ['' for j in range(len(kidney_visit_df))]
race_types = set()
for i in range(len(race_values)):
    these_vals = kidney_visit_df['race_source_value'].iloc[i].split('|')
    this_arr = []
    for x in range(len(these_vals)):
        this_val = these_vals[x].strip()
        if this_val != '':
            if this_val == 'White, non-Hispanic':
                this_val = 'White'
            if this_val == 'Other, Hispanic':
                this_val = 'Other'
            if this_val not in race_types:
                race_types.add(this_val)
            this_arr.append(this_val)
    race_values[i] = this_arr
race_types_list = list(race_types)
race_dummies = [[0 for i in range(len(race_types))] for j in range(len(kidney_visit_df))]
for i in range(len(kidney_visit_df)):
    this_val = race_values[i]
    for j in range(len(race_types_list)):
        if race_types_list[j] in this_val:
            race_dummies[i][j] = 1 
race_dummies = pd.DataFrame(race_dummies, columns = race_types_list)
kidney_visit_df = pd.concat([kidney_visit_df, race_dummies], axis = 1)


# code to create a dataframe on the patients who had kidney clinic visits and understand 
#who was getting a referral and how long it took them before they had a visit at the kidney clinic
unique_kidney_clinic_patients = kidney_visit_df['person_id'].unique()
diff_days = [0 for i in range(len(unique_kidney_clinic_patients))]
no_referral = [0 for i in range(len(unique_kidney_clinic_patients))]
kidney_clinic_date = [0 for i in range(len(unique_kidney_clinic_patients))]
visit_before_referral = [0 for i in range(len(unique_kidney_clinic_patients))]
proper_referral = [1 for i in range(len(unique_kidney_clinic_patients))]
for i in range(len(unique_kidney_clinic_patients)):
    
    person_referrals = kidney_referral_df[kidney_referral_df['person_id'] == unique_kidney_clinic_patients[i]]
    person_visits = kidney_visit_df[kidney_visit_df['person_id'] == unique_kidney_clinic_patients[i]]
    
    if len(person_referrals) > 0:
        these_visits = person_visits[person_visits['visit_time'] > person_referrals['first_referral'].iloc[0]]
        if len(these_visits) > 0: 
            kidney_clinic_date[i] = min(these_visits['visit_time'])
            diff_days[i] = (min(these_visits['visit_time']) -  person_referrals['first_referral'].iloc[0]).days
        else:
            visit_before_referral[i] = 1
    else:
        no_referral[i] = 1
        kidney_clinic_date[i] = min(person_visits['visit_time'])
    
    if no_referral[i] == 1 or visit_before_referral[i] == 1: 
        proper_referral[i] = 0


unique_kidney_clinic_df = pd.DataFrame(unique_kidney_clinic_patients, columns = ['person_id'])
unique_kidney_clinic_df['no_referral'] = pd.Series(no_referral, index = unique_kidney_clinic_df.index)
unique_kidney_clinic_df['visit_before_referral'] = pd.Series(visit_before_referral, index = unique_kidney_clinic_df.index)
unique_kidney_clinic_df['proper_referral'] = pd.Series(proper_referral, index = unique_kidney_clinic_df.index)
unique_kidney_clinic_df['kidney_clinic_date'] = pd.Series(kidney_clinic_date, index = unique_kidney_clinic_df.index)
unique_kidney_clinic_df['days_referral_to_visit'] = pd.Series(diff_days, index = unique_kidney_clinic_df.index)

#we need to offset dates by one month to make sure the quarter assignments align with the date of the 
#CKD-EPI 2021 implementation
unique_kidney_clinic_df['kidney_clinic_date'] = pd.to_datetime(unique_kidney_clinic_df['kidney_clinic_date'])
unique_kidney_clinic_df['kidney_clinic_date'] = unique_kidney_clinic_df['kidney_clinic_date'] + pd.DateOffset(months=1)
unique_kidney_clinic_df['year'] = unique_kidney_clinic_df['kidney_clinic_date'].dt.year
unique_kidney_clinic_df['quarter'] = unique_kidney_clinic_df['kidney_clinic_date'].dt.quarter


#we also need to offset this for the kidney clinic dataframe to ensure it aligns with the 
#CKD-EPI 2021 implementation date
kidney_visit_df['visit_time'] = kidney_visit_df['visit_time'] + pd.DateOffset(months=1)
kidney_visit_df['year'] = kidney_visit_df['visit_time'].dt.year
kidney_visit_df['quarter'] = kidney_visit_df['visit_time'].dt.quarter
kidney_visit_df_black = kidney_visit_df[kidney_visit_df['Black or African American'] == 1]
kidney_visit_df_nonblack = kidney_visit_df[kidney_visit_df['Black or African American'] == 0]


#we normalize the number of kidney clinic referrals by the number of overall visits at SHC 
#this is already date-shifted in SQL code
query="""
SELECT 
*
FROM`{work_project_id}.{work_dataset_id}.overall_visit_count` 
""".format_map({'cdm_project_id': cdm_project_id,
                'cdm_dataset_id': cdm_dataset_id, 
               'work_project_id': work_project_id,
                  'work_dataset_id': work_dataset_id})
overall_visit_df = client.query(query).to_dataframe()
overall_visit_df.head()
overall_visit_df.columns = ['year', 'quarter', 'visit_count']

#function to compute difference in months between two dates
def month_diff(a, b):
    return 12 * (a.year - b.year) + (a.month - b.month)

#creates the kidney referral dataframe aggregated by quarter

#we normalize all rates by 10,000 people
norm_num = 10000

kidney_visit_agg = kidney_visit_df.groupby(['year','quarter']).agg('nunique')['person_id'].reset_index()
kidney_visit_agg  = pd.merge(kidney_visit_agg , overall_visit_df, left_on = ['year', 'quarter'], 
        right_on = ['year', 'quarter'], how = 'left')

unique_kidney_clinic_df2 = unique_kidney_clinic_df[unique_kidney_clinic_df['proper_referral'] == 1]
kidney_clinic_df_agg  = unique_kidney_clinic_df2[['year', 'quarter', 
                                          'days_referral_to_visit']].groupby(['year', 'quarter']).agg('median')['days_referral_to_visit'].reset_index()

kidney_visit_agg  = pd.merge(kidney_visit_agg , kidney_clinic_df_agg, left_on = ['year', 'quarter'], 
        right_on = ['year', 'quarter'], how = 'left')
timestamp = [0 for i in range(len(kidney_visit_agg ))]
quarters = [1,2,3,4]
quarter_month = [12,3,6,9]

for i in range(len(kidney_visit_agg )):
    this_quarter = kidney_visit_agg['quarter'].iloc[i]
    quarter_index = quarters.index(this_quarter)
    if this_quarter == 1: 
        timestamp[i] = pd.Timestamp(kidney_visit_agg['year'].iloc[i] - 1, 
                                quarter_month[quarter_index], 1)
    else:
        timestamp[i] = pd.Timestamp(kidney_visit_agg['year'].iloc[i], 
                                quarter_month[quarter_index], 1)                            
                                
kidney_visit_agg['timestamp'] = pd.Series(timestamp, index = kidney_visit_agg.index)
kidney_visit_agg = kidney_visit_agg[(kidney_visit_agg['timestamp'] >= pd.Timestamp('2019-3-01'))]

#Since the end of our study happened in the middle of the quarter, we need to remove the last observation
kidney_visit_agg = kidney_visit_agg.iloc[:len(kidney_visit_agg) - 1]
kidney_visit_agg = kidney_visit_agg.reset_index()
kidney_visit_agg = kidney_visit_agg.drop(columns = ['index'])

kidney_visit_agg['range'] = pd.Series(range(len(kidney_visit_agg)), index = kidney_visit_agg.index)
kidney_visit_agg['time'] = kidney_visit_agg['range'] + 1
kidney_visit_agg['rate'] = kidney_visit_agg['person_id']*norm_num/kidney_visit_agg['visit_count']
kidney_visit_agg['rate'] = kidney_visit_agg['rate'].astype(float)
kidney_visit_agg['formula_change'] = [1 if kidney_visit_agg['timestamp'].iloc[x] >= pd.Timestamp('2021-12-01')
                                               else 0 for x in range(len(kidney_visit_agg))]

change_index = kidney_visit_agg[kidney_visit_agg['formula_change'] == 1].index[0]
formula_change = [0 for i in range(len(kidney_visit_agg))]
#assigns the temporary lag period (we want a nine-month temporary period with the full implementation at Sept 2022)
lag = 12
for i in range(len(kidney_visit_agg)):
    if kidney_visit_agg['timestamp'].iloc[i] >= pd.Timestamp('2021-9-01'):
        diff = month_diff(kidney_visit_agg['timestamp'].iloc[i],pd.Timestamp('2021-9-01')) 
        if diff < lag:
            formula_change[i] = diff/lag
        else:
            formula_change[i] = 1
kidney_visit_agg['formula_change_lag'] = pd.Series(formula_change, index = kidney_visit_agg.index)


kidney_visit_agg.columns = ['year', 'quarter', 'kidney_visit_count', 'visit_count', 
                               'days_referral_to_visit','timestamp', 'range', 'time',
                               'rate', 'formula_change','formula_change_lag']


#unadjusted 
columns_to_use =  ['time', 'formula_change_lag' ]
#adjusted 
columns_to_use =  ['time', 'formula_change_lag', 'days_referral_to_visit' ]

X = kidney_visit_agg[columns_to_use]
y = kidney_visit_agg['kidney_visit_count']
exog = sm.add_constant(X)
poission_model = sm.GLM(y, exog.astype(float),
                        offset = np.log(kidney_visit_agg['visit_count'].astype(float)),
    family=sm.families.Poisson())
poisson_result = poission_model.fit()
poisson_result.summary()

params = poisson_result.params
conf = poisson_result.conf_int()
conf['Rate Ratio'] = params
conf.columns = ['5%', '95%', 'Rate Ratio']
print(np.exp(conf))

#set up the counterfactual where CKD-EPI 2021 was not implemented
kidney_visit_agg_test = kidney_visit_agg.copy()
kidney_visit_agg_test['formula_change'] = [0 for i in range(len(kidney_visit_agg_test))]
kidney_visit_agg_test['formula_change_lag'] = [0 for i in range(len(kidney_visit_agg_test))]
X_test = kidney_visit_agg_test[columns_to_use]
exog_test = sm.add_constant(X_test)
prediction_r = poisson_result.get_prediction(exog.astype(float))
prediction_test_r = poisson_result.get_prediction(exog_test.astype(float))
kidney_visit_agg['predictions_poisson'] = (poisson_result.predict(exog.astype(float)))*norm_num
kidney_visit_agg['prediction_poisson_lower'] = (prediction_r.summary_frame(alpha = 0.05)['mean_ci_lower'])*norm_num
kidney_visit_agg['prediction_poisson_upper'] = (prediction_r.summary_frame(alpha = 0.05)['mean_ci_upper'])*norm_num
kidney_visit_agg['predictions_test_poisson'] = (poisson_result.predict(exog_test.astype(float)))*norm_num
kidney_visit_agg['predictions_test_lower'] = (prediction_test_r.summary_frame(alpha = 0.05)['mean_ci_lower'])*norm_num
kidney_visit_agg['predictions_test_upper'] = (prediction_test_r.summary_frame(alpha = 0.05)['mean_ci_upper'])*norm_num

#prediction with CKD-EPI 2021
print (kidney_visit_agg[kidney_visit_agg['formula_change'] == 1]['predictions_poisson'].mean())
print (kidney_visit_agg[kidney_visit_agg['formula_change'] == 1]['prediction_poisson_lower'].mean())
print (kidney_visit_agg[kidney_visit_agg['formula_change'] == 1]['prediction_poisson_upper'].mean())

#prediction without CKD-EPI 2021 (counterfactual)
print (kidney_visit_agg[kidney_visit_agg['formula_change'] == 1]['predictions_test_poisson'].mean())
print (kidney_visit_agg[kidney_visit_agg['formula_change'] == 1]['predictions_test_lower'].mean())
print (kidney_visit_agg[kidney_visit_agg['formula_change'] == 1]['predictions_test_upper'].mean())


#plot Figure 2 (overall cohort)
plt.figure(figsize = (15,10))
plt.rcParams.update({'font.size': 16})
plt.scatter(kidney_visit_agg['timestamp'],(kidney_visit_agg['kidney_visit_count']/kidney_visit_agg['visit_count'])*norm_num, linestyle='dashed', color = 'black')

kidney_visit_agg2 = kidney_visit_agg[kidney_visit_agg['timestamp'] >= '2021-12-01']
plt.plot(kidney_visit_agg2['timestamp'].values,kidney_visit_agg2['predictions_poisson'].values, color = 'red', label = 'CKD-EPI 2021')
plt.fill_between(kidney_visit_agg2['timestamp'].values,
         (kidney_visit_agg2['prediction_poisson_lower'].values), (kidney_visit_agg2['prediction_poisson_upper'].values),
                                                  color = 'red', alpha = 0.1)

plt.plot(kidney_visit_agg['timestamp'].values,kidney_visit_agg['predictions_test_poisson'].values,
         color = 'grey', label = 'Race-adjusted')
plt.fill_between(kidney_visit_agg['timestamp'].values,
         (kidney_visit_agg['predictions_test_lower'].values), (kidney_visit_agg['predictions_test_upper'].values),
                                                  color = 'grey', alpha = 0.1)

plt.text(pd.Timestamp('2020-07-15'), plt.gca().get_ylim()[1] - 4, 'CKD-EPI 2021 implementation',
         fontsize=16, color='black')

plt.axvline(x=pd.Timestamp(2021, 12,1), color = 'black')
plt.legend()
plt.ylabel('per 10,000 patients')
plt.xlabel('Year-month')



# ### Documented as Black or African American

#this is already date-shifted in SQL
query="""
 SELECT 
 *
 FROM`{work_project_id}.{work_dataset_id}.X` 
 """.format_map({'cdm_project_id': cdm_project_id,
                 'cdm_dataset_id': cdm_dataset_id, 
                'work_project_id': work_project_id,
                   'work_dataset_id': work_dataset_id})
black_visit_count_df = client.query(query).to_dataframe()
black_visit_count_df.head()
black_visit_count_df.columns = ['year', 'quarter', 'visit_count_black']


#set up the aggregated dataframe for those documented as Black or African American
kidney_visit_agg_black = kidney_visit_df_black.groupby(['year','quarter']).agg('nunique')['person_id'].reset_index()
kidney_visit_agg_black = pd.merge(kidney_visit_agg_black ,
                                      black_visit_count_df, left_on = ['year', 'quarter'], 
        right_on = ['year', 'quarter'], how = 'left')
kidney_visit_agg_black  = pd.merge(kidney_visit_agg_black , kidney_clinic_df_agg, left_on = ['year', 'quarter'], 
        right_on = ['year', 'quarter'], how = 'left')

timestamp = [0 for i in range(len(kidney_visit_agg_black ))]
for i in range(len(kidney_visit_agg_black )):
    this_quarter = kidney_visit_agg_black['quarter'].iloc[i]
    quarter_index = quarters.index(this_quarter)
    if this_quarter == 1: 
        timestamp[i] = pd.Timestamp(kidney_visit_agg_black['year'].iloc[i] - 1, 
                                quarter_month[quarter_index], 1)
    else:
        timestamp[i] = pd.Timestamp(kidney_visit_agg_black['year'].iloc[i], 
                                quarter_month[quarter_index], 1)                            
                                
kidney_visit_agg_black['timestamp'] = pd.Series(timestamp, index = kidney_visit_agg_black.index)
kidney_visit_agg_black = kidney_visit_agg_black[(kidney_visit_agg_black['timestamp'] >= pd.Timestamp('2019-3-01'))]

kidney_visit_agg_black = kidney_visit_agg_black.iloc[:len(kidney_visit_agg_black) - 1]
kidney_visit_agg_black = kidney_visit_agg_black.reset_index()
kidney_visit_agg_black = kidney_visit_agg_black.drop(columns = ['index'])
kidney_visit_agg_black['range'] = pd.Series(range(len(kidney_visit_agg_black)), index = kidney_visit_agg_black.index)
kidney_visit_agg_black['time'] = kidney_visit_agg_black['range'] + 1

kidney_visit_agg_black['rate'] = kidney_visit_agg_black['person_id']*norm_num/kidney_visit_agg_black['visit_count_black']

kidney_visit_agg_black['rate'] = kidney_visit_agg_black['rate'].astype(float)

kidney_visit_agg_black['formula_change'] = [1 if kidney_visit_agg_black['timestamp'].iloc[x] >= pd.Timestamp('2021-12-01')
                                               else 0 for x in range(len(kidney_visit_agg_black))]

change_index = kidney_visit_agg_black[kidney_visit_agg_black['formula_change'] == 1].index[0]
formula_change = [0 for i in range(len(kidney_visit_agg_black))]
lag = 12
for i in range(len(kidney_visit_agg_black)):
    if kidney_visit_agg_black['timestamp'].iloc[i] >= pd.Timestamp('2021-9-01'):
        diff = month_diff(kidney_visit_agg_black['timestamp'].iloc[i],pd.Timestamp('2021-9-01')) 
        if diff < lag:
            formula_change[i] = diff/lag
        else:
            formula_change[i] = 1
kidney_visit_agg_black['formula_change_lag'] = pd.Series(formula_change, index = kidney_visit_agg_black.index)


kidney_visit_agg_black.columns = ['year', 'quarter', 'kidney_visit_count', 'visit_count_black', 
                               'days_referral_to_visit','timestamp', 'range', 'time',
                               'rate', 'formula_change','formula_change_lag']


#unadjusted 
columns_to_use =  ['time', 'formula_change_lag']
#adjusted
columns_to_use =  ['time', 'formula_change_lag', 'days_referral_to_visit']
X = kidney_visit_agg_black[columns_to_use]
y = kidney_visit_agg_black['kidney_visit_count']
exog = sm.add_constant(X)
poission_model = sm.GLM(y, exog.astype(float),
                        offset = np.log(kidney_visit_agg_black['visit_count_black'].astype(float)),
    family=sm.families.Poisson())
poisson_result = poission_model.fit()
poisson_result.summary()


params = poisson_result.params
conf = poisson_result.conf_int()
conf['Rate Ratio'] = params
conf.columns = ['5%', '95%', 'Rate Ratio']
print(np.exp(conf))


kidney_visit_agg_test_black = kidney_visit_agg_black.copy()
kidney_visit_agg_test_black['formula_change'] = [0 for i in range(len(kidney_visit_agg_test_black))]
kidney_visit_agg_test_black['formula_change_lag'] = [0 for i in range(len(kidney_visit_agg_test_black))]
X_test = kidney_visit_agg_test_black[columns_to_use]
exog_test = sm.add_constant(X_test)
prediction_r = poisson_result.get_prediction(exog.astype(float))
prediction_test_r = poisson_result.get_prediction(exog_test.astype(float))
kidney_visit_agg_black['predictions_poisson'] = (poisson_result.predict(exog.astype(float)))*norm_num
kidney_visit_agg_black['prediction_poisson_lower'] = (prediction_r.summary_frame(alpha = 0.05)['mean_ci_lower'])*norm_num
kidney_visit_agg_black['prediction_poisson_upper'] = (prediction_r.summary_frame(alpha = 0.05)['mean_ci_upper'])*norm_num
kidney_visit_agg_black['predictions_test_poisson'] = (poisson_result.predict(exog_test.astype(float)))*norm_num
kidney_visit_agg_black['predictions_test_lower'] = (prediction_test_r.summary_frame(alpha = 0.05)['mean_ci_lower'])*norm_num
kidney_visit_agg_black['predictions_test_upper'] = (prediction_test_r.summary_frame(alpha = 0.05)['mean_ci_upper'])*norm_num

#with CKD-EPI 2021
print (kidney_visit_agg_black[kidney_visit_agg_black['formula_change'] == 1]['predictions_poisson'].mean())
print (kidney_visit_agg_black[kidney_visit_agg_black['formula_change'] == 1]['prediction_poisson_lower'].mean())
print (kidney_visit_agg_black[kidney_visit_agg_black['formula_change'] == 1]['prediction_poisson_upper'].mean())


#without CKD-EPI 2021
print (kidney_visit_agg_black[kidney_visit_agg_black['formula_change'] == 1]['predictions_test_poisson'].mean())
print (kidney_visit_agg_black[kidney_visit_agg_black['formula_change'] == 1]['predictions_test_lower'].mean())
print (kidney_visit_agg_black[kidney_visit_agg_black['formula_change'] == 1]['predictions_test_upper'].mean())


#plot Figure 2 (documented as Black or African American)
plt.figure(figsize = (15,10))
plt.scatter(kidney_visit_agg_black['timestamp'],(kidney_visit_agg_black['kidney_visit_count']/kidney_visit_agg_black['visit_count_black'])*norm_num, color = 'black')
kidney_visit_agg_black2 = kidney_visit_agg_black[kidney_visit_agg_black['timestamp'] >= '2021-12-01']

plt.plot(kidney_visit_agg_black2['timestamp'].values,kidney_visit_agg_black2['predictions_poisson'].values, color = 'red', label = 'CKD-EPI 2021')
plt.fill_between(kidney_visit_agg_black2['timestamp'].values,
         (kidney_visit_agg_black2['prediction_poisson_lower'].values), (kidney_visit_agg_black2['prediction_poisson_upper'].values),
                                                  color = 'red', alpha = 0.1)

plt.plot(kidney_visit_agg_black['timestamp'].values,kidney_visit_agg_black['predictions_test_poisson'].values,
         color = 'grey', label = 'Race-adjusted')
plt.fill_between(kidney_visit_agg_black['timestamp'].values,
         (kidney_visit_agg_black['predictions_test_lower'].values), (kidney_visit_agg_black['predictions_test_upper'].values),
                                                  color = 'grey',alpha = 0.1)

plt.text(pd.Timestamp('2020-07-15'), plt.gca().get_ylim()[1] - 10, 'CKD-EPI 2021 implementation',
         fontsize=16, color='black')

plt.axvline(x=pd.Timestamp(2021, 12,1), color = 'black')
plt.legend()
plt.ylabel('per 10,000 patients')
plt.xlabel('Year-month')


# ### Not documented as Black or African American

#already date shifted in SQL

query="""
 SELECT 
 *
 FROM`{work_project_id}.{work_dataset_id}.X` 
 """.format_map({'cdm_project_id': cdm_project_id,
                 'cdm_dataset_id': cdm_dataset_id, 
                'work_project_id': work_project_id,
                   'work_dataset_id': work_dataset_id})
nonblack_visit_count_df = client.query(query).to_dataframe()
nonblack_visit_count_df.head()
nonblack_visit_count_df.columns = ['year', 'quarter', 'visit_count_nonblack']


kidney_visit_agg_nonblack = kidney_visit_df_nonblack.groupby(['year','quarter']).agg('nunique')['person_id'].reset_index()

kidney_visit_agg_nonblack  = pd.merge(kidney_visit_agg_nonblack ,
                                      nonblack_visit_count_df, left_on = ['year', 'quarter'], 
        right_on = ['year', 'quarter'], how = 'left')

kidney_visit_agg_nonblack  = pd.merge(kidney_visit_agg_nonblack , kidney_clinic_df_agg, left_on = ['year', 'quarter'], 
        right_on = ['year', 'quarter'], how = 'left')


timestamp = [0 for i in range(len(kidney_visit_agg_nonblack ))]
for i in range(len(kidney_visit_agg_nonblack )):
    this_quarter = kidney_visit_agg_nonblack['quarter'].iloc[i]
    quarter_index = quarters.index(this_quarter)
    if this_quarter == 1: 
        timestamp[i] = pd.Timestamp(kidney_visit_agg_nonblack['year'].iloc[i] - 1, 
                                quarter_month[quarter_index], 1)
    else:
        timestamp[i] = pd.Timestamp(kidney_visit_agg_nonblack['year'].iloc[i], 
                                quarter_month[quarter_index], 1)                            
                                
kidney_visit_agg_nonblack['timestamp'] = pd.Series(timestamp, index = kidney_visit_agg_nonblack.index)
kidney_visit_agg_nonblack = kidney_visit_agg_nonblack[(kidney_visit_agg_nonblack['timestamp'] >= pd.Timestamp('2019-3-01'))]

kidney_visit_agg_nonblack  = kidney_visit_agg_nonblack .iloc[:len(kidney_visit_agg_nonblack ) - 1]
kidney_visit_agg_nonblack = kidney_visit_agg_nonblack.reset_index()
kidney_visit_agg_nonblack = kidney_visit_agg_nonblack.drop(columns = ['index'])
kidney_visit_agg_nonblack['range'] = pd.Series(range(len(kidney_visit_agg_nonblack)), index = kidney_visit_agg_nonblack.index)
kidney_visit_agg_nonblack['time'] = kidney_visit_agg_nonblack['range'] + 1

kidney_visit_agg_nonblack['rate'] = kidney_visit_agg_nonblack['person_id']*norm_num/kidney_visit_agg_nonblack['visit_count_nonblack']

kidney_visit_agg_nonblack['rate'] = kidney_visit_agg_nonblack['rate'].astype(float)

kidney_visit_agg_nonblack['formula_change'] = [1 if kidney_visit_agg_nonblack['timestamp'].iloc[x] >= pd.Timestamp('2021-12-01')
                                               else 0 for x in range(len(kidney_visit_agg_nonblack))]

change_index = kidney_visit_agg_nonblack[kidney_visit_agg_nonblack['formula_change'] == 1].index[0]

formula_change = [0 for i in range(len(kidney_visit_agg_nonblack))]
lag = 12
for i in range(len(kidney_visit_agg_nonblack)):
    if kidney_visit_agg_nonblack['timestamp'].iloc[i] >= pd.Timestamp('2021-9-01'):
        diff = month_diff(kidney_visit_agg_nonblack['timestamp'].iloc[i],pd.Timestamp('2021-9-01')) 
        if diff < lag:
            formula_change[i] = diff/lag
        else:
            formula_change[i] = 1
kidney_visit_agg_nonblack['formula_change_lag'] = pd.Series(formula_change, index = kidney_visit_agg_nonblack.index)


kidney_visit_agg_nonblack.columns = ['year', 'quarter',
                                     'kidney_visit_count',
                                     'visit_count_nonblack',
                               'days_referral_to_visit','timestamp', 'range', 'time',
                               'rate', 'formula_change','formula_change_lag']


#unadjusted 
columns_to_use =  ['time', 'formula_change_lag']
#adjusted
columns_to_use =  ['time', 'formula_change_lag', 'days_referral_to_visit']
X = kidney_visit_agg_nonblack[columns_to_use]
y = kidney_visit_agg_nonblack['kidney_visit_count']
exog = sm.add_constant(X)
poission_model = sm.GLM(y, exog.astype(float),
                        offset = np.log(kidney_visit_agg['visit_count'].astype(float)),
    family=sm.families.Poisson())
poisson_result = poission_model.fit()
poisson_result.summary()

params = poisson_result.params
conf = poisson_result.conf_int()
conf['Rate Ratio'] = params
conf.columns = ['5%', '95%', 'Rate Ratio']
print(np.exp(conf))

kidney_visit_agg_test_nonblack = kidney_visit_agg_nonblack.copy()
kidney_visit_agg_test_nonblack['formula_change'] = [0 for i in range(len(kidney_visit_agg_test_nonblack))]
kidney_visit_agg_test_nonblack['formula_change_lag'] = [0 for i in range(len(kidney_visit_agg_test_nonblack))]
kidney_visit_agg_test_nonblack['time:FC'] = [0 for i in range(len(kidney_visit_agg_test_nonblack))]
X_test = kidney_visit_agg_test_nonblack[columns_to_use]
exog_test = sm.add_constant(X_test)
prediction_r = poisson_result.get_prediction(exog.astype(float))
prediction_test_r = poisson_result.get_prediction(exog_test.astype(float))
kidney_visit_agg_nonblack['predictions_poisson'] = (poisson_result.predict(exog.astype(float)))*norm_num
kidney_visit_agg_nonblack['prediction_poisson_lower'] = (prediction_r.summary_frame(alpha = 0.05)['mean_ci_lower'])*norm_num
kidney_visit_agg_nonblack['prediction_poisson_upper'] = (prediction_r.summary_frame(alpha = 0.05)['mean_ci_upper'])*norm_num
kidney_visit_agg_nonblack['predictions_test_poisson'] = (poisson_result.predict(exog_test.astype(float)))*norm_num
kidney_visit_agg_nonblack['predictions_test_lower'] = (prediction_test_r.summary_frame(alpha = 0.05)['mean_ci_lower'])*norm_num
kidney_visit_agg_nonblack['predictions_test_upper'] = (prediction_test_r.summary_frame(alpha = 0.05)['mean_ci_upper'])*norm_num

#with CKD-EPI 2021
print (kidney_visit_agg_nonblack[kidney_visit_agg_nonblack['formula_change'] == 1]['predictions_poisson'].mean())
print (kidney_visit_agg_nonblack[kidney_visit_agg_nonblack['formula_change'] == 1]['prediction_poisson_lower'].mean())
print (kidney_visit_agg_nonblack[kidney_visit_agg_nonblack['formula_change'] == 1]['prediction_poisson_upper'].mean())
#without CKD-EPI 2021
print (kidney_visit_agg_nonblack[kidney_visit_agg_nonblack['formula_change'] == 1]['predictions_test_poisson'].mean())
print (kidney_visit_agg_nonblack[kidney_visit_agg_nonblack['formula_change'] == 1]['predictions_test_lower'].mean())
print (kidney_visit_agg_nonblack[kidney_visit_agg_nonblack['formula_change'] == 1]['predictions_test_upper'].mean())


#Plot Figure 2 (not documented as Black or African American)
plt.figure(figsize = (15,10))
plt.scatter(kidney_visit_agg_nonblack['timestamp'],(kidney_visit_agg_nonblack['kidney_visit_count']/kidney_visit_agg_nonblack['visit_count_nonblack'])*norm_num, color = 'black')

kidney_visit_agg_nonblack2 =kidney_visit_agg_nonblack[kidney_visit_agg_nonblack['timestamp'] >= '2021-12-01']

plt.plot(kidney_visit_agg_nonblack2['timestamp'].values,kidney_visit_agg_nonblack2['predictions_poisson'].values, color = 'red', label = 'CKD-EPI 2021')
plt.fill_between(kidney_visit_agg_nonblack2['timestamp'].values,
         (kidney_visit_agg_nonblack2['prediction_poisson_lower'].values), (kidney_visit_agg_nonblack2['prediction_poisson_upper'].values),
                                                  color = 'red', alpha = 0.1)

plt.plot(kidney_visit_agg_nonblack['timestamp'].values,kidney_visit_agg_nonblack['predictions_test_poisson'].values,
         color = 'grey', label = 'Race-adjusted')
plt.fill_between(kidney_visit_agg_nonblack['timestamp'].values,
         (kidney_visit_agg_nonblack['predictions_test_lower'].values), (kidney_visit_agg_nonblack['predictions_test_upper'].values),
                                                  color = 'grey', alpha = 0.1)


plt.text(pd.Timestamp('2020-07-15'), plt.gca().get_ylim()[1] - 5, 'CKD-EPI 2021 implementation',
         fontsize=16, color='black')


plt.axvline(x=pd.Timestamp(2021, 12,1), color = 'black')
plt.legend()
plt.ylabel('per 10,000 patients')
plt.xlabel('Year-month')





