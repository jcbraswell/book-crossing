WITH POPULATION AS (
SELECT DISTINCT
    UGIDX_CHT_EMPLID AS EMPLID,
    UGIDX_CHT_TERM AS DEM_COHORT,
   (UGIDX_DEG_TERM-UGIDX_CHT_TERM) AS DEM_DIFF_INDX

FROM IRAMASTER.SSD_UGIDX_MR
WHERE UGIDX_STUDENT_TYPE='FTF' AND UGIDX_CHT_TERM in ('2094','2104','2114','2124')
),

PELL AS (
SELECT 
    PELLTOT_EMPLID,
    PELLTOT_ELIG_TYPE AS PELL_ELIGIBILITY

FROM IRAMASTER.SSD_PELL_SR
WHERE PELLTOT_EMPLID IS NOT NULL
),

DAEONE AS(
SELECT DISTINCT * FROM(
    SELECT 
        DAE_EMPLID,
        DAE_SEX_CODE AS GENDER,
        MIN(DAE_RACE_ETH) AS ETHNICITY,
        MIN(DAE_MIN_STATUS) AS MINORITY
    FROM IRAMASTER.SSD_DAE_SR 
    WHERE DAE_EMPLID IS NOT NULL
    GROUP BY DAE_EMPLID,DAE_SEX_CODE)
),

DAETWO AS(
SELECT DISTINCT * FROM(
    SELECT 
        DAE_EMPLID,
        MIN(DAE_FIRST_GEN) AS FIRST_GENERATION
    FROM IRAMASTER.SSD_DAE_SR 
    WHERE DAE_EMPLID IS NOT NULL
    GROUP BY DAE_EMPLID)
),

DAETHREE AS (
SELECT DISTINCT * FROM(
    SELECT 
        DAE_EMPLID,
        MAX(CASE WHEN DAE_DEP_FAMILY LIKE '%99%' THEN -1 ELSE DAE_DEP_FAMILY END) AS DEP_FAMILY_SIZE,
        MAX(CASE WHEN DAE_INDEP_FAMILY LIKE '%99%' THEN -1 ELSE DAE_INDEP_FAMILY END) AS APPLICANT_FAMILY_SIZE,
        MAX(DAE_INDEP_INCOME) AS APPICANT_INCOME
    FROM IRAMASTER.SSD_DAE_SR 
    WHERE DAE_EMPLID IS NOT NULL
    GROUP BY DAE_EMPLID)
),

DEMOGRAPHICS AS (
    SELECT 
        T1.DAE_EMPLID,
        GENDER,
        ETHNICITY,
        FIRST_GENERATION,
        CASE WHEN DEP_FAMILY_SIZE LIKE '%-1%' THEN 'NA' ELSE TO_CHAR(DEP_FAMILY_SIZE) END AS DEP_FAMILY_SIZE,
        MINORITY,
        CASE WHEN APPLICANT_FAMILY_SIZE LIKE '%-1%' THEN 'NA' ELSE TO_CHAR(APPLICANT_FAMILY_SIZE) END AS APPLICANT_FAMILY_SIZE,
        CASE 
            WHEN APPICANT_INCOME=1 THEN 'LESS THEN $6000'
            WHEN APPICANT_INCOME=2 THEN '$6,000 TO $11,999'
            WHEN APPICANT_INCOME=3 THEN '$12,000 TO $23,999'
            WHEN APPICANT_INCOME=4 THEN '$24,000 TO $35,999'
            WHEN APPICANT_INCOME=5 THEN '$36,000 TO $47,999'
            WHEN APPICANT_INCOME=6 THEN '$48,000 TO $59,999'
            WHEN APPICANT_INCOME=7 THEN '$60,000 OR MORE'
            WHEN APPICANT_INCOME=8 THEN 'CANNOT ESTIMATE'
            WHEN APPICANT_INCOME=9 THEN 'NO RESPONSE'
    END  AS APPLICANT_INCOME
    FROM DAEONE T1
    LEFT JOIN DAETWO T2 ON T1.DAE_EMPLID=T2.DAE_EMPLID
    LEFT JOIN DAETHREE T3 ON T1.DAE_EMPLID=T3.DAE_EMPLID
),

ADMISSIONS AS (
SELECT 
    ESA_EMPLID,
    CASE WHEN ESA_ACT_COMPOSITE = 0 THEN NULL ELSE ESA_ACT_COMPOSITE END AS ACT_COMP,
    CASE WHEN ESA_ACT_READING = 0 THEN NULL ELSE ESA_ACT_READING END AS ACT_READ,
    CASE WHEN ESA_ACT_MATH = 0 THEN NULL ELSE ESA_ACT_MATH END AS ACT_MATH,
    CASE WHEN ESA_ACT_ENGLISH = 0 THEN NULL ELSE ESA_ACT_ENGLISH END AS ACT_ENG, 
    CASE WHEN ESA_ACT_SCI_REASONING = 0 THEN NULL ELSE ESA_ACT_SCI_REASONING END AS ACT_SCI, 
    CASE WHEN ESA_SAT_CRIT_READING = 0 THEN NULL ELSE ESA_SAT_CRIT_READING END AS SAT_READ, 
    CASE WHEN ESA_SAT_MATH = 0 THEN NULL ELSE ESA_SAT_MATH END AS SAT_MATH, 
    CASE WHEN ESA_SAT_COMP_SCORE = 0 THEN NULL ELSE ESA_SAT_COMP_SCORE END AS SAT_COMP,
    CASE WHEN ESA_HS_GPA = 0 THEN NULL ELSE ESA_HS_GPA END AS GPA_HS,
    NULL AS AP

FROM  IRAMASTER.SSD_ESA_SR
LEFT JOIN DEMOGRAPHICS ON ESA_EMPLID=DAE_EMPLID
LEFT JOIN PELL ON ESA_EMPLID=PELLTOT_EMPLID

),

FINAL_JOIN AS (
SELECT DISTINCT
  *  
FROM
            POPULATION              T0
LEFT  JOIN  DEMOGRAPHICS            T1 ON T0.EMPLID           = T1.DAE_EMPLID 
LEFT  JOIN  PELL                    T2 ON T1.DAE_EMPLID       = T2.PELLTOT_EMPLID
LEFT  JOIN  ADMISSIONS              T3 ON T1.DAE_EMPLID       = T3.ESA_EMPLID
  
) 

SELECT *
FROM 
  FINAL_JOIN
