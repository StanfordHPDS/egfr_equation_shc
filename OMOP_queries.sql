-- Identify the first referral for each patient with referrals 
-- The concept id for nephrology referrals may change according to data refreshes. You will need to contact
--the STARR OMOP team if this happens, and you cannot identify the corresponding concept_id.


CREATE OR REPLACE table `X.first_referral` as 
select distinct t.person_id, t.first_referral, count(distinct vo1.visit_occurrence_id) as visit_count
FROM 
(SELECT person_id, min(o.observation_DATETIME) as first_referral
FROM `X.observation` o
 JOIN `X.concept` c 
  on o.observation_concept_id = c.concept_id 
  where c.concept_id in (4141568) -- this is the nephrology referral id that may change
group by person_id) as t 
join `X.visit_occurrence` vo1 
on vo1.person_id = t.person_id and vo1.visit_start_DATETIME < t.first_referral
group by t.person_id, t.first_referral;

-- Run person table, which identifies age, sex, race
CREATE OR REPLACE TABLE `X.person_table` AS
SELECT person_id, p.gender_concept_id, c1.concept_name as gender, year_of_birth, month_of_birth, day_of_birth, 
birth_DATETIME, p.race_source_value, c.concept_name as race, p.ethnicity_source_value, 
c2.concept_name as ethnicity FROM
 `X.person` p
 JOIN `X.concept` c 
 on p.race_concept_id = c.concept_id 
  JOIN `X.concept` c1 
 on p.gender_concept_id = c1.concept_id 
 JOIN `X.concept` c2 
 on p.ethnicity_concept_id= c2.concept_id;

--Join the kidney referral table with the person table to get all of the patient demographics
CREATE OR REPLACE table `X.first_referral_full` AS 
select fr.person_id, fr.first_referral, fr.visit_count, 
gender_concept_id, gender, year_of_birth, month_of_birth, day_of_birth, 
birth_DATETIME, race_source_value,  race, ethnicity_source_value,  ethnicity from 
`X.first_referral` fr 
join `X.person_table` pt
on fr.person_id = pt.person_id;


-- The concept ids for kidney clinic sites may change according to data refreshes. You will need to contact
--the STARR OMOP team if this happens, and you cannot identify the corresponding concept_ids.
--Identify the kidney clinic visits that occurred at SHC 
CREATE OR REPLACE TABLE `X.kidney_clinic_visits` AS
SELECT vo.person_id, vo.visit_occurrence_id, vo.visit_start_DATETIME as visit_time
FROM `X.visit_occurrence` vo
JOIN `X.care_site` cs 
on vo.care_site_id = cs.care_site_id
where cs.care_site_id in (3292,3223, 4616,4187, 5877, 4360, 5021, 6243, 4453, 4808, 4943, 5854);

--Join with the person table
CREATE OR REPLACE table `X.kidney_clinic_visits_full` AS 
select fr.person_id, fr.visit_occurrence_id, fr.visit_time, 
gender_concept_id, gender, year_of_birth, month_of_birth, day_of_birth, 
birth_DATETIME, race_source_value,  race, ethnicity_source_value,  ethnicity from 
`X.kidney_clinic_visits` fr 
join `X.person_table` pt
on fr.person_id = pt.person_id;


--normalize by tables 
--overall visit count 
CREATE OR REPLACE table `X.overall_visit_count` as 
SELECT EXTRACT(YEAR from t.visit_start) as YEAR, EXTRACT(QUARTER from t.visit_start) as QUARTER,  count(distinct t.person_id) as visit_count
from (select DATE_ADD(visit_start_DATETIME, INTERVAL 1 MONTH) as visit_start,person_id from
 `X.visit_occurrence`) t
 group by EXTRACT(YEAR from t.visit_start), EXTRACT(QUARTER from t.visit_start);

--Black or African American
CREATE OR REPLACE table `X.overall_visit_Black` AS
SELECT EXTRACT(YEAR from t.visit_start) as YEAR, EXTRACT(QUARTER from t.visit_start) as QUARTER, count(distinct t.person_id) as visit_count from 
(select p.person_id, DATE_ADD(visit_start_DATETIME, INTERVAL 1 MONTH) as visit_start from 
`X.visit_occurrence` vo
join `X.person` p 
on p.person_id = vo.person_id
where race_source_value like '%Black or African American%') t
group by EXTRACT(YEAR from t.visit_start), EXTRACT(QUARTER from t.visit_start);

--not Black or African American
CREATE OR REPLACE table `X.overall_visit_NonBlack` AS
SELECT EXTRACT(YEAR from t.visit_start) as YEAR, EXTRACT(QUARTER from t.visit_start) as QUARTER, count(distinct t.person_id) as visit_count from 
(select p.person_id, DATE_ADD(visit_start_DATETIME, INTERVAL 1 MONTH) as visit_start from 
`X.visit_occurrence` vo
join `X.person` p 
on p.person_id = vo.person_id
where race_source_value not like '%Black or African American%') t
group by EXTRACT(YEAR from t.visit_start), EXTRACT(QUARTER from t.visit_start);





