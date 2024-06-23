# MRI Imaging Planes

Reference
* https://radiology.ucsf.edu/patient-care/for-patients/video/ucsf-radiology-how-read-images

![image](https://user-images.githubusercontent.com/5284312/233124806-20005748-7480-4ee3-adc1-1e8623c46438.png)

* Axial - image is from bottom up. So left side of body is on right

## Anatomical Coordinate System

The most important model coordinate system for medical imaging techniques is the anatomical space (also called patient coordinate system). This space consists of three planes to describe the standard anatomical position of a human:

* the axial plane is parallel to the ground and separates the head (Superior) from the feet (Inferior)
* the coronal plane is perpendicular to the ground and separates the front from (Anterior) the back (Posterior)
* the sagittal plane separates the Left from the Right

From these planes it follows that all axes have their notation in a positive direction (e.g. the negative Superior axis is represented by the Inferior axis).

The anatomical coordinate system is a continuous three-dimensional space in which an image has been sampled. In neuroimaging, it is common to define this space with respect to the human whose brain is being scanned. Hence the 3D basis is defined along the anatomical axes of anterior-posterior, inferior-superior, and left-right.

However different medical applications use different definitions of this 3D basis. Most common are the following bases:

* LPS (Left, Posterior, Superior) is used in DICOM images and by the ITK toolkit
    * from right towards left
    * from anterior towards posterior
    * from inferior towards superior
* RAS (Right, Anterior, Superior) is similar to LPS with the first two axes flipped and used by 3D Slicer
    * from left towards right
    * from posterior towarsd anterior
    * from inferior towards superior

https://nipy.org/nibabel/image_orientation.html
