# Deep-Learning Convolutional Neural Networks Accurately Classify Genetic Mutations in Gliomas

## References
* https://www.ajnr.org/content/early/2018/05/10/ajnr.A5667 (Authored by Peter Chang)

## Abstract
#### Purpose
To train a convolutional neural network to independently predict underlying molecular genetic mutation status in gliomas with high accuracy and identify the most predictive imaging features for each mutation.
#### Materials and Methods
* MRI images of 259 patients with either low or high-grade gliomas
* Train CNN to classify
    1. isocitrate dehydrogenase 1 (IDH1) mutation status
    2. 1p/19q codeletion
    3. O6-methylguanine-DNA methyltransferase (MGMT) promotor methylation status
#### Results
* Classification accuracy of each method
    1. IDH1 mutation status, 94%
    2. 1p/19q codeletion, 92%
    3.  MGMT promotor methylation status 83%
* Each genetic category was also associated with distinctive imaging features such as definition of tumor margins, T1 and FLAIR suppression, extent of edema, extent of necrosis, and textural features.
#### Conclusion
* It's possible to classify individual genetic mutations from both low and high grade gliomas.

## Introduction
* Diffuse infiltrating gliomas are a heterogeneous group of primary tumors with highly variable imaging characteristics, response to therapy, clinical course, and prognoses.
* During tumor development (tumorigenesis) muliple variations of genetic and epigenetic mutations occur. Some mutations confer improved survivorship and/or favorable response to therapies (radiation, chemotherapy, etc). 
* Biopsies are currently required to classify the genetic or molecular alterations in the glioma.
    * Drawbacks
        * Biopsies are often limited to the easily accessible areas of the enhancing tumor.
        * Molecular genetic testing is costly and slow
### Known MR Imaging features of gliomas
* spatial and temporal variations in genetic expression are known to result in heterogeneous alterations in tumor biologytumors are heterogeneous.
* changes in angiogenesis, cellular proliferation, cellular invasion, and apoptosis
* varying degrees of enhancement, infiltration, hemorrhage, reduced diffusion, edema, and necrosis
* Visually AcceSAble Rembrandt Images (VASARI) feature set. Attempt to standardize the visual interpretation of malignant gliomas for tissue classification. It's a rule-based lexicon to improve the reproducibility of interpretation. But this approach requires a priori feature selection and human visual interpretation, which might difficult to distill given the large scale of the MRI image and variations of the tumors.

## Materials and Methods
### Subjects
* Retrospectively obtained from The Cancer Imaging Archives for patients with either low- or high-grade gliomas
* Corresponding molecular genetic information from The Cancer Genome Atlas
* Only patients with full preoperative MR imaging, including T2, FLAIR, and T1-weighted pre- and postcontrast acquisitions, were included in the analysis.
### Image Preprocessing
* Imaging modalities coregistered with FMRIB Linear Image Registration Tool (FLIRT; http://www.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT). Most often the post-contrast T1-weighted were used as reference volume.
* Coregistered input was normalized using z-score values (u=0, sigma=1). 
* apply 3D CNN to rermove extracranial structures
* apply automated brain tumor segmentation tool to remove lesion margins (Best method from 2016 Multimodal Brain Tumor Segmentation Challenge. CNN to perform whole-tumor segmentation masks). Crop the image to the tumor region. 
* Resize tumor crop to 32x32x4, where (I think the 4 channels are the T2, FLAIR, T1-pre, T1-post).
### Feature Analysis
* Apply PCA to the 64d featurer vectors
### Statistical Analysis
* Measure algorithm accuracy by AUC score
* 5-fold cross-validation. 20% of dataset held out for validation.

## Results
### Subjects
* 5259 axial slices of tumor from 259 patients with gliomas (135 men/ 122 women / 2 unknown, 53.2
years; mean survival, 18.8 year)
* glioma grades:
    * 21.2% (55/259) grade II
    * 22.8% (59/259) grade III
    * 55.2% (143/259) grade IV
* IDH1 mutant and wild-type tumors accounted for 45.9% (119/259) and 54.1% (140/259) of patients, respectively
*  1p/19q codeletions and nondeletions accounted for 12.0% (31/259) and 88.0% (228/259) of patients, respectively
*  MGMT promoter methylated and unmethylated accounted for 56.4% (146/259) and 43.6% (113/259) of patients, respectively
*  Mean tumor size 105.6 cm3
### CNN Accuracy
* IDH1 mutation 
    * accuracy: (mean, 94%; range between cross validations, 90%–96%)
    * AUC: (mean, 0.91; range, 0.89 –0.92)
* 1p/19q codeletion
    * accuracy: (mean, 92%; range, 88%–95%)
    * AUC:  (mean, 0.88; range, 0.85–0.90)
* MGMT promoter methylation
    * accuracy: (mean, 83%; range, 76%–88%)
    * AUC: (mean, 0.81; range, 0.76 –0.84)
### CNN Training Schedule
* 25k iterations (~3k epochs)
* batch size: 12-48

### Feature Analysis
* IDH mutation features
* Ip19 codeleation features

## Identification of prototypical features corrrelating with MGMT Promoter Methylation

Positive (methylated) features
* nodular and heterogeneous enhancement
* masslike FLAIR edema (larger lesions with a higher portion of nonenhancing tumor component) with cortical invovlement
* presence of eccentric cyst
* slight frontal and superficial temporal predominance  (which aligns with existing literature which suggests that the tumors have a frontal lobe predominance). 

Negative (unmethylated) features
* thick enhancement with central necrosis
* thin rim enhancement
* solid enhancement
* ill-defined margins
* infiltrative edema patterns.
* slight, deep, temporal predominance

![image](https://user-images.githubusercontent.com/5284312/232322794-717c0ca1-d587-4a09-ad48-10a9d945fe1a.png)

* This approach of leveraging the DNN to characterize the MGMT features can be used to guide practicing physicians.

## Limitations

### Risk of Overfitting
Overfitting could be an issue due to small dataset (n=259)

To mitigate this risk
* Light-weight DNN. Small number of parameters and layers and high normalization
* Downsample images to 32x32x4
### The dataset is heterogenous from multiple hospitals
* The Cancer Imaging Archives dataset is a heterogeneousdataset from multiple different contributing sites
* Heterogeneity of the data could hinder DNN accuracy, but in this case the DNN still perforrmed well
### Lack of independent dataset
* The "validation" dataset was sampled from the primary dataset, e.g. via 5-fold cross validation. It would be preferred to evaluate the model on an unseen dataset. 

## Conclusion
The results of our study show the feasibility of a deep-learning CNN approach for the accurate classification of individual genetic mutations of both low- and high-grade gliomas. Furthermore, we emonstrate that the relevant MR imaging features acquired from an added dimensionality-reduction technique are concordant with existing literature, showing that neural networks are capable of learning key imaging components without prior feature selection or human directed training.
