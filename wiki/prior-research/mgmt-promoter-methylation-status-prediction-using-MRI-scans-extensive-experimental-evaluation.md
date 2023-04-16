# MGMT promoter methylation status prediction using MRI scans? An extensive experimental evaluation of deep learning models

## References
* https://arxiv.org/pdf/2304.00774.pdf
### Conclusion
The authors conclude that there is no correlation between the brain MRIs and methylation status. 
## Biology background
* MGMT is a DNA repair enzyme that will reduce the effect of chemotherapy drug temozolomide (TMZ) which is an alkylating agent that tries to keep the cell from reproducing by attaching to guanine DNA, inducing cell apoptosis. 
* If MGMT promoter is methylated, then MGMT enzyme is not expressed and patients will respond well to TMZ.
* Survival rate: Patients with methylated promoter have median survival of 21.7 months, as opposed to 15.3 months for unmethylated MGMT promoters. (Hegi et al 2005) https://www.nejm.org/doi/full/10.1056/nejmoa043331 
* Clinical study of 334 patients concludes that there’s no connection between methylation of MGMT promoter and overall survival rate.They could not predict response of TMZ in the cohort even after epigenetic silencing of MGMT promoter. https://pubmed.ncbi.nlm.nih.gov/33116181/  
* Another Study concludes that MRI imaging features cannot be used to noninvasively predict MGMT promoter methylation status because MGMT status is not related to the phenotypical biology in IDH wild-type GBMs. https://pubmed.ncbi.nlm.nih.gov/32688383/
    * They evaluate the relationship between MGMT status and phenotypical tumor biology. Look for 22 histopathological features, immunohistochemical proliferation index, microvessel density measurements, conventional magnetic resonance imaging characteristics, preoperative speed of tumor growth, and overall survival. None of the investigated histological or radiological features were significantly associated with MGMT status. 
    * Also conclude that the MGMT status is not correlated with a less aggressive tumor biology.
## Prior Work
Prior work struggles to achieve high AUC (or classification accuracy) on the more diverse, comprehensive dataset

0.62 AUC was best score

## Dataset
* 585 patients in dataset. 307 methylated. 278 nonmethylated. 
* Variable slice thickness from 0.43 to 6mm
* Different number of slices per patient. 
* Different scan orientations - coronal, axial, sagittal
* They use the Task 1 (segmentation) dataset as well because it’s already aligned
## Results
* All experiments achieved AUC in range 0.5682 to 0.6178, which isn’t very good… so not doing good job of classifying. Cannot differentiate between methylated status of tumor patients.
### Task 2 dataset
* Apply inhouse preprocessing techniques
* Resample the scans to the same axial plane for consistency
* Extract same number of slices per patient (following example from winner of Kaggle competition)
* Only train on single modality - T1wCE, FLAIR
### Task 1 dataset
* Combine T1wCE, FLAIR, T2 modalities
* AUC still is in range of 0.536 - 0.631, so no model does well. 
### Custom models
* 2 stems - one for FLAIR one for T1wCE, then concat or add features together before making the classification. Only 0.63 at best
* Segmentation mask to extract only tumor region (as ROI) in a cropped slice-by-slice fashion with size 32x32, also tried 128x128. -> only get around 0.53 AUC
* Contrastive learning-based SSL pretraining didn’t improve.
## Error Analysis
### Grad-CAM and Occlusion Sensitivity
The model can localize the tumor region despite incorrect final prediction. 
The model detects abnormalities (it leverages information in the tumor region to make final prediction).

Despite successfully localizing the tumor, the model cannot detect features within the tumor that could help discriminate MGMT promoter methylation status
<img src=artifacts/ml-eval-salency.png width=70% height=70%>
### Feature Maps of CNN
Even in the final layers, the positive/negative samples are entangled indicating that the model cannot find features that differentiate between the classes. However, the features look disentangled when the same model is trained to predict the malignancy of lung nodules. 
<img src=artifacts/ml-eval-feature-maps.png width=70% height=70%>
### Probability Distribution of the Predictions
* A well trained binary classifier should have a bimodal prediction probability distribution (probs close to 0 or 1)
* The model ends up having a unimodal distribution, so it’s confused

<img src=artifacts/ml-eval-prob-dist.png width=70% height=70%>
### Loss Landscape
* BCE loss function
* Random state = loss = 0.69
* The model gets loss = 0.70, which means it’s still confused

<img src=artifacts/ml-eval-loss-landscape.png width=70% height=70%>

## Limitations to Generalizability
* Previous positive performance were likely because of limited dataset size and lack of independent dataset to test on.
* Suggest that it might be able to predict methylation status by combining other biomarkers or prognostic factors….
* Importance of thorough and unbiased validation for future studies and clinical implementations.
## Recommendations
1. Collect data from a diverse range of patients and tumors. 
    * Different tumor properties - sizes, shapes, characteristics
    * Different ages, genders, races, ethnicities
2. Use external validation dataset
    * Evaluate model on dataset from a cohort not part of the training dataset
3. Conduct explainability analysis
    * To evaluate performance of the model and identify any biases or limitations in performance. Try to understand which factors are most important for the model’s predictions and identify any trends or patterns in the data that could drive the model’s decisions.
4. Engage with clinicians and other stakeholders
    * By engaging with clinicians and other stakeholders, it will be possible to develop deep learning models that are more closely aligned with the needs and goals of these users and can be more easily integrated into clinical practice
5. Use multi-modal data
    * Structured data - electronic medical records
    * Unstructured data - radiology images
    * A more holistic view of patient
6. Develop standardized protocols and evaluation metrics
    * Consistent objective
7. Reproducibility
    * Important for determining model validity and reliability
    * Establish trust in scientific findings
