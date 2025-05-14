## Brain Tumor Radiomics: MGMT Promoter Classification from MRI

This project focused on classifying the **MGMT promoter methylation status** of glioblastoma brain tumors using 3D MRI scans. MGMT promoter methylation status is a key genetic marker that determines whether a patient is likely to respond to chemotherapy. Traditionally, this status is identified through a biopsy—this project aimed to predict it **non-invasively** from imaging data using deep learning.

This project was performed in collaboration with [Peter Chang](https://www.faculty.uci.edu/profile/?facultyId=6569) and the [Center for Artificial Intelligence in Diagnostic Medicine](https://www.caidm.som.uci.edu/).

### Key Highlights
- **Model:** 3D convolutional neural network (CNN) with deep supervision at multiple intermediate feature scales.
- **Objective:** Classify tumors based on MGMT promoter methylation status.
- **Input:** Multimodal MRI with 4 channels:
  - T2-weighted
  - FLAIR
  - T1-weighted (pre-contrast)
  - T1-weighted (post-contrast)
- **Clinical relevance:** Enables faster, less expensive diagnosis without invasive biopsy.

## Multimodal MRI Visualization
MRI sequences
  - T2-weighted
  - FLAIR
  - T1-weighted (pre-contrast)
  - T1-weighted (post-contrast)

MRI anatomical planes
- Sagittal
- Coronal
- Axial

![image](https://github.com/user-attachments/assets/2a772d76-4e56-4927-bc67-5a4bf62ca57f)



### Preprocessing Pipeline
- Co-registration of all MRI sequences to a common coordinate space (axial, coronal, sagittal).
- Skull-stripping to isolate the brain region.
- Resampling to standard 3D volume size and voxel spacing.
- Z-score normalization of image intensities.

### Data Sampling and Augmentation
Due to the relatively small dataset (~500 patients), early experiments experienced significant overfitting to the training data. To mitigate this, I implemented a 3D patch sampling strategy along with extensive data augmentation. Patches were sampled from each MRI brain volume based on a probability map designed to favor regions within the brain—especially areas containing tumor tissue. This approach increased the likelihood of extracting patches that included tumor regions. I also experimented with various 3D patch sizes to optimize performance.

- Data augmentation:
  - Random affine transformations
  - Random gamma correction
  - Random bias field
  - Random motion blur
- Applied 3D tumor segmentation masks to exclude irrelevant anatomy.

### Visualization & Tooling
- Developed tools to:
  - Display 2D slices from 3D volumes
  - Visualize tumor segmentation labels
  - Output scan metadata and classification predictions

### Challenges
- Small dataset (~500 samples) with significant heterogeneity across hospitals.
- Difficult to interpret model predictions due to complex visual features not easily recognized by non-radiologists.
- High intra- and inter-patient variability in MRI appearance.
- Training on 3D data across four MRI modalities led to slow runtimes and significant memory limitations. These constraints made it difficult to train large models using high-resolution inputs, necessitating the use of smaller 3D patches during training.

## Dataset Visualization
- visualization of the full dataset form the axial view across all 4 MRI modalities
- the displayed slice of the 3D MRI volume is the tumor "center of mass" slice
- the yellow dot is the "center of mass" X,Y coordinate

![tumor-center-mass](https://github.com/user-attachments/assets/b5026b4e-49ed-4752-8bef-928f7d0f66bd)

### Technologies Used
- [PyTorch Lightning](https://www.lightning.ai/)
- [MONAI (Medical Open Network for AI)](https://monai.io/)
- [TorchIO](https://torchio.readthedocs.io/)

## Installation 
```
# create virtual environment
pip install -e .[dev,notebook]
```
### Jupyter notebook development
```
jupyter nbextension install itkwidgets --user --py
jupyter nbextension enable itkwidgets --user --py

jupyter notebook
```
### Tensorboard
```
tensorboard --logdir=/path/to/logs --port 6006
```
