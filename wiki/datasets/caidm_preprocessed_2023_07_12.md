# CAIDM Preprocessed RSNA-MICCAI BRATS 2021 dataset

## 3D 240x240 spatial resolution preprocessed
* Dataset prepared by Peter Chang on 07/12/2023
* https://www.dropbox.com/s/sz9nnl4ug9u7hjz/miccai_rsna_raw_nii.tar?dl=0
### Dataset Description
* optimized for 3D modeling including aggressive subsampling. For the radiogenomics task
* Each volume is resampled to (155, 240, 240) per the BRATS standard space, and aligned to each other in the axial orientation. All the original raw data is of course various different shapes and resolutions, but this will be as close to a standardized high resolution volume that we can expect for MRI.

The data is from an older Nifti (*.nii.gz) format. You'll need the nibabel library to open (https://nipy.org/nibabel/) 

### Dataset Inspection
* 661 folders total
* 76 with `MGMT-` prefix
* 585 with `P-` prefix
Each folder contains 5 files
`fla.nii.gz`, `seg.nii.gz`, `t1c.nii.gz`, `t1w.nii.gz`, `t2w.nii.gz`

### Tensors
* Image tensors such as `fla`, `t1c`, `t1w`, `t2w` contain unnormalized pixel values, e.g. the values might range from 0 - 2934.
* Segmentation mask tensors `seg` contain pixel values: 0, 1, 2, 4. These correspond to the official tumor segmentation labels given in the competition

### Tumor segmentation labels
1. Gd-enhanced tumor (ET - label 4).
    - Gd = gadolinium
    - “Enhanced tumor”
    Visually avid as well as faint enhancement on T1Gd MRI.
2. Peritumoral edematous/invaded tissue (ED - label 2).
    - Hyperintensive signal envelope on T2 FLAIR volumes that includes infiltrated non enhanced tumor as well as vasogenic edema in the peritumoral region.
3. Necrotic tumor core (NCR - label 1).
    - Hypointensive on T1Gd MRI
4. “Tumor core”. TC. 
    - Includes ET and NCR
    - Typically this is what is surgically removed
5. “Whole tumor” WT
    - Includes TC and ED

- The challenge evaluated segmentations of “enhancing tumor” ET, “tumor core” TC and “whole tumor” WT
    ```
    Labels:
    1 = NCR
    2 = ED
    4 = ET
    0 = everything else

    ET = 4
    TC = 4 + 1
    WT = 4 + 1 + 2
    ```