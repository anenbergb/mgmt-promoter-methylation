# CAIDM Preprocessed RSNA-MICCAI BRATS 2021 dataset

## 3D 96x96 spatial resolution preprocessed
* Dataset prepared by Peter Chang on 04/14/2023
* https://www.dropbox.com/s/bj9t3n5c3lcb8pu/data.npz?dl=0
### Dataset Description
* optimized for 3D modeling including aggressive subsampling. For the radiogenomics task


### Preprocessing:
For the sake of easy model experimentation, the following data preprocessing steps have been performed on the raw MRI data. Raw data is available and we can work to preprocess as needed for additional evaluation.
1. Co-registration of all sequences to each other (same coordinate space)
2. Brain extraction (removal of all non-brain structures)
3. Resampling to uniform 3D matrix shape of (48, 96, 96)
4. Z-score normalization of data clipped to a range of [-4, +4]

### Advantages to subsampling
It's hypothesized that the downsampled dataset will yield better results including decreased overfitting (compared to high resolution 2D images)

1. We do not have a lot of data, just ~500+ patients. This is by far the biggest problem for medical deep learning projects. As a result, we typically cannot build very large models unless we rely on some other pretraining strategy (but this in itself is problematic as there are no good large pretraining cohorts for medical data and generalization from ImageNet or the equivalent is limited). Using full resolution images is more likely to overfit. As a result, for this problem, I recommend subsampling the original data volume.
2. The use of high resolution data would preclude a full 3D model. In my opinion, the location of the tumor in 3D space (e.g., relative position within the brain) will actually be a useful piece of information to retain. A 2D model will have difficultly incorporating 3D context.
3. Reduce impact of intra-hospital variance. Part of the overfitting with a multi-institutional dataset like this is that each hospital's own machines / imaging protocols result in slight differences in image quality, noise, etc. These differences are much more likely to be reduced at lower resolutions; thus subsampling also helps to reduce intra-hospital variance and improve model generalization.


All data is archived as a single compressed Numpy (*.npz) archive with the following key-value pairs.
* Inputs: Four MRI sequences for each of N=565 patients. 
* All 3D volumes are of shape (N, Z, Y, X, 1) = (565, 48, 96, 96, 1).
```
t2w: T2-weighted MRI of shape (565, 48, 96, 96, 1)
fla: FLAIR-weighted MRI of shape (565, 48, 96, 96, 1)
t1w: T1 -weighted pre-contrast MRI of shape (565, 48, 96, 96, 1)
t1c: T1-weighted post-contrast MRI of shape (565, 48, 96, 96, 1)
```
For model input, consider using any individual sequence and/or multiple sequences stacked along the channels dimension. In my preliminary work, the “t2w” and “t1c” sequences provide the most “signal” for radiogenomics task while the "fla" sequence provide the most "signal" for the segmentation task.

Outputs (target): Binary outcome for MGMT methylation status.
* lbl: binary vector of shape (565, 1) of MGMT methylation status (radiogenomics task)
* tum: binary segmentation tumor mask of shape (565, 48, 96, 96, 1) (segmentation task although this information will also likely be very useful as adjunctive data for the radiogenomics task)
```
# --- Load data
data = np.load('./data.npz')

# --- Create Numpy array for t2w data
t2w = np.array(data['t2w'])

# --- Create Numpy array for lbl data
lbl = np.array(data['lbl'])
```
