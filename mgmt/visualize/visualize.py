import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from monai.transforms.utils import rescale_array
from monai.visualize import blend_images
from PIL import Image

# image = blend_images(
#     # convert from BDHWC to HWC CHW where C=1, and D=48 3D depth channels
#     image=np.transpose(data[modality][patient_i, s, ...], axes=(2, 0, 1)),
#     label=np.transpose(data["tum"][patient_i, s, ...], axes=(2, 0, 1)),
#     alpha=0.25,
#     cmap="hsv",
#     rescale_arrays=True,
# )


def plot_classification_grid(preds: np.ndarray, target: np.ndarray, patient_id: np.ndarray) -> np.ndarray:
    assert len(preds) == len(target)
    assert len(preds) == len(patient_id)

    errors = np.abs(preds - target)
    errors_bucketed = np.digitize(errors, np.arange(0, 1.1, 0.1)) - 1  # 10 buckets
    zipped = zip(patient_id, errors_bucketed, target)
    zipped = sorted(zipped, key=lambda x: x[0])

    width = int(np.ceil(np.sqrt(len(preds))))
    grid = np.zeros((width, width), dtype=int)
    cmap = colors.LinearSegmentedColormap.from_list("Custom", ["green", "white", "red"], N=10)
    fig, ax = plt.subplots()
    for i, (pid, error_bucket, tar) in enumerate(zipped):
        grid[i // width, i % width] = error_bucket
        methyl = "(M)" if tar == 1 else ""
        text = f"{pid}{methyl}"
        ax.text(i % width, i // width, text, ha="center", va="center", color="black")
    ax.imshow(grid, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def add_color_border(img, border_width=10, color="green"):
    """
    Assume img is shape (H,W,C)
    """
    assert isinstance(img, np.ndarray)
    height = img.shape[0]
    width = img.shape[1]
    frame_height = 2 * border_width + height
    frame_width = 2 * border_width + width
    framed_img = Image.new("RGB", (frame_width, frame_height), color)
    framed_img = np.array(framed_img)
    framed_img[border_width:-border_width, border_width:-border_width] = img
    return framed_img


def segmentation_label_color(labels=[0, 1, 2, 4], cmap="hsv", rgb_255=True):
    _cmap = cm.get_cmap(cmap)

    labels_np = np.array(labels)
    label_np = rescale_array(labels_np)
    label_rgb = _cmap(label_np)
    if rgb_255:
        colors = [tuple((255.0 * l).astype(np.uint8)[:3]) for l in label_rgb]
    else:
        colors = [tuple(l[:3]) for l in label_rgb]
    return colors
