from monai.visualize import blend_images

image = blend_images(
    # convert from BDHWC to HWC CHW where C=1, and D=48 3D depth channels
    image=np.transpose(data[modality][patient_i, s, ...], axes=(2, 0, 1)),
    label=np.transpose(data["tum"][patient_i, s, ...], axes=(2, 0, 1)),
    alpha=0.25,
    cmap="hsv",
    rescale_arrays=True,
)
