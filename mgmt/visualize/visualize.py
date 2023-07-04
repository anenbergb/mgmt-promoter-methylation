import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from monai.visualize import blend_images

# image = blend_images(
#     # convert from BDHWC to HWC CHW where C=1, and D=48 3D depth channels
#     image=np.transpose(data[modality][patient_i, s, ...], axes=(2, 0, 1)),
#     label=np.transpose(data["tum"][patient_i, s, ...], axes=(2, 0, 1)),
#     alpha=0.25,
#     cmap="hsv",
#     rescale_arrays=True,
# )


def plot_classification_grid(binary_preds: np.ndarray, target: np.ndarray, patient_id: np.ndarray) -> np.ndarray:
    assert len(binary_preds) == len(target)
    assert len(binary_preds) == len(patient_id)

    zipped = zip(binary_preds, target, patient_id)
    zipped = sorted(zipped, key=lambda x: x[2])

    width = int(np.ceil(np.sqrt(len(binary_preds))))
    grid = np.zeros((width, width), dtype=int)
    cmap = colors.ListedColormap(["red", "green"])
    fig, ax = plt.subplots()
    for i, (binary_pred, tar, pid) in enumerate(zipped):
        grid[i % width, i // width] = 1 if binary_pred == tar else 0
        methyl = "(M)" if tar == 1 else ""
        text = f"{pid}{methyl}"
        ax.text(i % width, i // width, text, ha="center", va="center", color="w")

    ax.imshow(grid, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
