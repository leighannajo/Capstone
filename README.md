
# Hospital Readmission Analysis

An analysis project by: Leighanna Hooper

**The data analysis and modeling can be located within my attached notebooks in this repository.  
They have been labeled and organized for your reading pleasure!

<img src=images/revolvingdoor.jpeg style="width:100%">


**Abstract:** <br>
Healthcare industry Providers understand that high hospital readmission rates are dangerous and even fatal for at risk patients. But excessive readmission rates can also threaten a hospital's financial health. Readmissions are already one of the costliest episodes to treat, with hospital costs reaching $41.3 billion for patients readmitted within 30 days of discharge. The financial burden of hospital readmissions also recently increased when, in 2013 under the Affordable Care Act (ACA), the Centers for Medicare and Medicaid implemented programs designed to penalize hospitals for “excess” readmissions in comparison to “expected” readmissions. The Readmissions Reduction Program penalizes hospitals up to 3% of their total Medicare payments for high rates of patients readmitted within 30 days of the original discharge.

My focus for this project was to learn, from previous patient hospital encounters, whether readmissions can be avoided by tagging high-risk patients. By analyzing these past encounters, and associating certain demographics, diagnoses or discharge dispositions, healthcare providers may be able to minimize the readmission rates within their hospital. Using these patient encounter characteristics that were shown to be related to high readmission rates, I built a machine learning model to assist even further in predicting encounters that are at high risk for readmission.

Since there are many pieces of information belonging to a patient encounter, health care data has many complications and limitations to be aware of when analyzing. I established the level of the patient data and used aggregations to avoid unnecessary noise in the data as well as to avoid unwanted biases. The main identifying factors analyzed were patient demographics (age, gender, race), primary diagnosis and patient discharge disposition for the encounter. Analysis findings show that age, circulatory (or cardiac) and respiratory primary diagnoses and discharges to home (without home health care) and facility transfers are the highest predictors for excess readmission. Modeling predictions had accuracy of 58%, with 90% recall True Positive predictions of predicting a patient would be readmitted and they were and under 10% predictions of a patient not being readmitted yet they actually were. With cost-benefit analysis in mind, I believe this is a significant starting point for further evaluation and analysis.

Cost-benefit analysis used: Variable costs per day of hospital (hospital incurred):

**The average cost of a patient's first day is $1246.** <br>

**The average cost of a patient's last day $304.**<br>

**When a patient is readmitted the hospital, they incur this first day cost twice, whereas an extra day is financially more beneficial.** <br>

<a href="https://www.journalacs.org/article/S1072-7515(00)00352-5/fulltext https://www.beckershospitalreview.com/finance/average-cost-per-inpatient-day-across-50-states.html">"Average Cost per Inpatient Day Across 50 States" - Beckers Hospital Review</a>

I would recommend that hospital healthcare providers engage in more rigorous discharge planning when patients are being discharged to their homes. It would also be advantageous for inpatient hospital providers of high-risk patients to be provided more in-depth education of their health issues along with resources to assist in further patient education. Lastly, hospitals, insurance companies and providers would benefit from patient incentives for regular physical exams and preventative care treatment with their outpatient primary care physicians. This project has a lot of further research planned, including model tuning to lower the False Positive rate of predicting that a patient will be readmitted and they are actually not high risk for readmission and deeper analysis of patient encounter data available. This project would benefit from collection of more data from a larger range of hospitals or locations and collection of supplementary features of patient medical encounter data.


**Data Description:** <br>
Due to healthcare PHI regulations (HIPAA, HITECH), there are limited number of publicly available datasets and some datasets require training and approval. So, for this project I am using a dataset from UC Irvine.

<a href="https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008">UC Irvine Dataset</a>

<img src=images/data.png style="width:100%">

</details>
<br>
<img src=style="width:50%">

## Motivation / Objective

**Objective:** <br>
This project will demonstrate the importance of building the right data representation at the encounter level, with appropriate filtering and preprocessing/feature engineering of key medical code sets. This project will also analyze and interpret a model for biases across key demographic groups.

**Context:** <br>
EHR data is becoming a key source of real-world evidence (RWE) for the pharmaceutical industry and regulators to make decisions on clinical trials.




## Analysis:

The main identifying factors that were analyzed were patient demographics, the diagnoses with the highest patient readmission rate as well where or who the patient is discharged to, for instance home, transferred to a physical rehab or home with home health.


Readmissions in the data are not easily separatable.

<img src=images/pca.png style="width:100%">


<details>
<summary style="font-size:4.5vw">Demographics </summary>
<br>

Gender was the first demographic explored and female patients account for 10% more readmissions than male patients, now this could be biased due to females typically seeking medical attention more frequently than males.

Race was also analyzed and Caucasian and African American patients were represented the highest.  This could also be biased based on location of the hospitals in the data.  

<img src=images/demos.png style="width:100%">

Age groups, however, seemed to have a pretty good distribution across groups.  I found that patients in the age of 50-90 are at higher risk for readmission.



</details>

<details>
<summary style="font-size:4.5vw">Primary Diagnosis</summary>
<br>

Circulatory/cardiac diagnoses account for the highest rates of readmission.

<img src=images/diagnoses.png  style="width:50%">
</details>

<img src=images/cardioreadmissions.jpeg  style="width:50%">

<details>
<summary style="font-size:4.5vw">Discharge Disposition</summary>
<br>
Discharge disposition was analyzed and 
A patient’s discharge to home is clearly has highest rate of readmissions with facility transfers, which could be anything from PT rehab, skilled nursing facilities, group homes, etc  also has very significant readmission rate




<img src=images/discharges.png  style="width:50%">
</details>
</details>

Something interesting that caught my attention and is worth more analysis is that, on average, patients that are being transferred actually tend to have longer stays…which doesn’t really align with readmissions usually stemming from too short of a stay or an underestimation of possible treatment and care at home!  However, home discharges having a shorter stay does align with the high readmission rate theory.


<img src=images/timeanddischarge.png style="width:100%">

## Conclusions and Recommendations:

<img src=images/toolstoreduce.png style="width:50%">

Patient home discharge (without home healthcare) is the leading disposition related to readmissions usually due to underestimations made about the extent of a patient’s or family member’s ability to correctly follow discharge instructions or follow-up care as well as the push of administration to quickly discharge patients. I would recommend that hospital healthcare providers engage in more rigorous discharge planning when patients are being discharged to their homes. The discharge planning should start earlier (beginning of encounter) and include family members so that no information is lost and no assumptions are made. A more in-depth protocol for follow-up care with actual patient check-ins made, or trigger patient for non-compliance and attempt deeper check-in methods.

It would be advantageous for inpatient hospital providers of high-risk patients to be provided with more in-depth education of their health issues along with resources to assist in further patient education. We are aware from analysis that patients in the age range of 50-90, have a diagnosis of circulatory (cardiac) disease, respiratory disease or one related to digestive issues, trauma, genitourinary or musculoskeletal diseases and are discharged to home or another facility are at the highest risk of readmission, so these are the patients we would want to spend more time with to assess their understanding of their current health situation. We would want to provide as much assistance in guiding these patients to being including in their wellness and health.

Hospitals, insurance companies and providers would benefit from patient incentives for regular physical exams and preventative care treatment with their outpatient primary care physicians. Insurance companies already give incentives in the form of gift cards or waiving copay fees for physicals, but these can be pushed further for high-risk patients. These providers can make it more enticing for patients to be involved in and actively participate in their health and well-being.

<img src=images/penalties.jpeg style="width:80%">

## Limitations and Future Work: 
Limitations and biases are difficult to avoid with medical data, so being aware of them is important.

How can we account for these biases with future work?

- Collect more data: time between discharge and readmission.
- Collect more demographics across larger regions.
- Collect cardiac subset diagnoses.


## Appendix:
Using Aequitas to Check for Biases:

- To check for significant bias in the model cross race and gender:
- Plot metrics that are important for patient selection (with race and gender).
- Check for significant bias in the model across any of the groups.
- Check for bias across diagnoses.

<img src=images/Picture1.png style="width:80%">



**The data analysis and modeling can be located within my attached notebooks in this repository.  
They have been labeled and organized for your reading pleasure!





<a href= PatientReadmissions.pdf >Project Presentation Slides PDF</a>

<a href= https://youtu.be/Ihog__Rp1to >Project Presentation Video Blog</a>

<a href= https://leighannajohooper.medium.com/predicting-hospital-readmissions-part-1-af5ca97753cb >Project Presentation Blog</a>