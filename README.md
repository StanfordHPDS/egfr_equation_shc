# eGFR_equation_SHC
This repository contains the statistical/analytic code for our paper titled "Evaluating nephrology referrals and visits after removing race adjustment from estimated glomerular filtration rate (eGFR) equations" by Marika Cusick, Glenn Chertow, Doug Owens, Michelle Williams, and Sherri Rose. 

While the statistical/analytical code is made available, our data is generated from Stanford Health Care electronic health record data that includes patient identifiers and protected health information. These data cannot be accessed without approval from the Stanford Institutional Review Board. We have replaced all references to data tables with 'X'. 

Our code is written in SQL (BigQuery) and python version 3.9.1. Required python packages include pandas (version 2.0.3), google.cloud (3.3.5), matplotlib (3.3.3), numpy (version 1.25.1), and statsmodels (0.14.0). 

1. Create tables for analysis
Code: OMOP_queries.sql runs the SQL code to generate tables required for our analysis. Data from SHC is in the Observational Medical Outcomes Partnership (OMOP) common data model format. 

3. Analysis of nephrology referrals
Code: kidney_clinic_referral_1_31_24.py runs the interrupted time series regression to assess changes to the quarterly rates of nephrology referrals at SHC after the implementation of the eGFR formula without race adjustment (CKD-EPI 2021). 

4. Analysis of nephrology visits
Code: kidney_clinic_visit_1_31_24.py runs the interrupted time series regression to assess changes to the quarterly rates of nephrology visits at SHC after the implementation of the eGFR formula without race adjustment (CKD-EPI 2021). 



