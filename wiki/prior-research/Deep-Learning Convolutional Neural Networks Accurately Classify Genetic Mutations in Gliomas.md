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

![image](https://user-images.githubusercontent.com/5284312/232322794-717c0ca1-d587-4a09-ad48-10a9d945fe1a.png)
