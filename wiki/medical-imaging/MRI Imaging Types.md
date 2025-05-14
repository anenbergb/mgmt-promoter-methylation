# MRI Imaging Types
References
* https://case.edu/med/neurology/NR/MRI%20Basics.htm

## Basics of MRI
* Apply external magnetic field to align H20 in tissue (e.g. the H+)
* The alignment (magnetization) is perturbed/distrupted by an external Radio Field (RF). 
* The H+ nucli will emit RF energy as they relax from aligned state to resting state
* Fourier transform to convert from frequency to intensity
* Repetition Time (TR) - amount of time between successive pulse sequences applied to the same slice
* Time to Echo (TE) - time between delivery of RF pulse and receipt of echo signal

## Comparing MRI Scan Types

| Feature / Purpose                         | FLAIR                             | T1w                                | T1Gd                               | T2                                 |
|------------------------------------------|-----------------------------------|------------------------------------|------------------------------------|-------------------------------------|
| **Full Name**                            | Fluid Attenuated Inversion Recovery | T1-weighted (pre-contrast)         | T1-weighted with Gadolinium        | T2-weighted                        |
| **Primary Use**                          | Highlight lesions near CSF         | Structural anatomy, brain detail  | Detect blood-brain barrier disruption | Pathology, fluid detection        |
| **CSF Appearance**                       | **Dark** (suppressed)              | Dark                               | Dark                               | **Bright**                         |
| **Gray Matter Appearance**               | Intermediate                       | Gray                               | Gray                               | Lighter gray                       |
| **White Matter Appearance**              | Darker than gray matter            | Bright                             | Bright                             | Darker than gray matter            |
| **Lesions/Edema**                        | **Bright**                         | Iso- or hypointense                | Enhanced if contrast uptake        | **Bright**                         |
| **Tumors**                               | Bright if associated with edema    | May appear dark                    | **Enhance with contrast**          | Bright with associated edema       |
| **Multiple Sclerosis Plaques**           | **Easily visible** (periventricular) | Poorly seen                        | Enhanced if active                 | Seen but less distinct than FLAIR  |
| **Ischemia/Stroke**                      | Subacute: hyperintense             | Early infarct hard to detect       | Enhances subacute infarct edges   | Acute infarct appears bright       |
| **Hemorrhage**                           | Poor for early detection           | Early blood may be hyperintense    | Enhancing rim (if subacute)        | Variable signal based on stage     |
| **Contrast Agent Used**                  | No                                 | No                                 | **Yes (Gadolinium)**               | No                                 |
| **Key Strength**                         | Detecting lesions near CSF, MS     | High-resolution anatomy            | Visualizing contrast-enhancing lesions | Detecting edema, fluid, pathology |
| **CSF Suppression**                      | **Yes**                            | No                                 | No                                 | No                                 |


## T1 vs. T2 relaxation times
* T1 (longitudinal relaxation time)
  *  time constant which determines the rate at which excited protons return to equilibrium. 
  *  Measure of the time taken for spinning protons to realign with the external magnetic field
* T2 (transverse relaxation time)
  * time constant which determines the rate at which excited protons reach equilibrium or go out of phase with each other
  * Measure of the time taken for spinning protons to lose phase coherence among the nuclei spinning perpendicular to the main field.

## T1-weighted images
* short TE
* short TR
The contrast and brightness in the tissue is determined by the T1 properties of the tissue
## T2-weighted images
* longer TE
* longer TR

The contrast and brightness in the tissue is determined by the T2 properties of the tissue
## FLAIR (Fluid Attenuated Inversion Recovery)
similar to T2-weighted
* very long TE
* very long TR


Abnormalities remain bright but normal CSF fluid is attenuated and made dark.

|                 | TR (msec) | TE (msec) |
| --------------- | --------- | --------- |
| T1-Weighted (short TR & TE) | 500 | 14 |
| T2-Weighted (long TR & TE) | 4000 | 90 |
| Flair (very long TR & TE) | 9000 | 114 |

## T1-weighted + post contrast
* Infuse patient with Gadolinium (Gad) to provide contrast
* Gad changes signal intensities by shortening T1. Gad is very bright on T1-weighted image.
* Gad enhanced images are especially useful in looking at vascular structures and breakdown in the blood-brain barrier [e.g., tumors, abscesses, inflammation (herpes simplex encephalitis, multiple sclerosis, etc.)]

## Diffusion Weighted Imaging (DWI)
* designed to detect random movements of H20 protons
* Water diffuses freely through extracellular space and is stricted in intracellular space
* Diffusion is reduced in ischemic brain tissue.
  * During ischemia, the sodium - potassium pump shuts down and sodium accumulates intracellularly. 
  * High Na+ concentration intracellularly -> osmotic gradient -> Water moves inside the cell
  * At equilibrium the rate of water movement will decrease -> very bright DWI signal
* Very sensitive method for detecting acute stroke

## T1 vs. T2 vs. Flair
![image](https://user-images.githubusercontent.com/5284312/233119630-ed8d82c9-0aa3-4af5-a336-9b2307558128.png)

| Tissue | T1-Weighted | T2-Weighted | Flair |
| ----- | ------ | ---- | --- |
| CSF | Dark | Bright | Dark |
| White Matter | Light | Dark Gray | Dark Gray |
| Cortex | Gray | Light Gray | Light Gray | 
| Fat (within bone marrow) | Bright | Light | Light | 
| Inflammation (infection, demyelination) | Dark | Bright | Bright |

## T1 vs. T1 with Contrast
![image](https://user-images.githubusercontent.com/5284312/233120342-5a1f2c74-e34d-431a-b424-686aa8d593c2.png)
* Gad/contrast enhanced bright signal in the blood vessels

## Flair vs. Diffusion-weighted
![image](https://user-images.githubusercontent.com/5284312/233120605-0e1a5366-c0d0-4d1d-b186-83c363009736.png)
* Accute infarction only seen on DWI

## T1 vs. T2 on spine
![image](https://user-images.githubusercontent.com/5284312/233120702-d2295013-b1bd-4abb-9fa0-d077aa3a8812.png)

| Tissue | T1-Weighted | T2-Weighted | 
| --- | ---| --- |
| CSF | Dark | Bright | 
| Muscle | Gray | Dark Gray | 
| Spinal Cord | Gray | Dark Gray |
| Fat (subcutaneous tissue) | Bright | Light |
| Disk (if intact & hydrated) | Gray | Bright |
| Air (pharynx) | Very Dark | Very Dark |
| Inflammation (edema, infarction, demyelination) | Dark | Bright |

## Limitations of MRI
* Subject to motion artifact
* Inferior to CT in detecting acute hemorrhage
* Inferior to CT in detection of bony injury
* Requires prolonged acquisition time for many images
