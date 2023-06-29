import io

import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
from torchio.transforms.preprocessing.spatial.to_canonical import ToCanonical
from torchio.visualization import color_labels, rotate


def plot_volume(
    image: tio.Image,
    radiological=True,
    channel=-1,  # default to foreground for binary maps
    axes=None,
    cmap=None,
    output_path=None,
    show=True,
    xlabels=True,
    percentiles=(0.5, 99.5),
    figsize=None,
    reorient=True,
    indices=None,
    single_axis=None,  # one of (Sagittal, Coronal, Axial)
):
    fig = None

    axes_names = "sagittal", "coronal", "axial"
    if single_axis is not None:
        assert single_axis in ("sagittal", "coronal", "axial")
        axes_names = (single_axis,)
    num_rows = len(axes_names)

    if axes is None:
        fig, axes = plt.subplots(1, num_rows, figsize=figsize, squeeze=False)

    # sag_axis, cor_axis, axi_axis = axes

    if reorient:
        image = ToCanonical()(image)  # type: ignore[assignment]
    data = image.data[channel]
    if indices is None:
        indices = np.array(data.shape) // 2
    i, j, k = indices
    slice_x = rotate(data[i, :, :], radiological=radiological)
    slice_y = rotate(data[:, j, :], radiological=radiological)
    slice_z = rotate(data[:, :, k], radiological=radiological)
    kwargs = {}
    is_label = isinstance(image, tio.LabelMap)
    if isinstance(cmap, dict):
        slices = slice_x, slice_y, slice_z
        slice_x, slice_y, slice_z = color_labels(slices, cmap)
    else:
        if cmap is None:
            cmap = "cubehelix" if is_label else "gray"
        kwargs["cmap"] = cmap
    if is_label:
        kwargs["interpolation"] = "none"

    sr, sa, ss = image.spacing
    kwargs["origin"] = "lower"

    if percentiles is not None and not is_label:
        p1, p2 = np.percentile(data, percentiles)
        kwargs["vmin"] = p1
        kwargs["vmax"] = p2

    axes_map = {
        "sagittal": {"aspect": ss / sa, "slice": slice_x, "xlabel": "A", "ylabel": "S", "title": "Sagittal"},
        "coronal": {"aspect": ss / sr, "slice": slice_y, "xlabel": "R", "ylabel": "S", "title": "Coronal"},
        "axial": {"aspect": sa / sr, "slice": slice_z, "xlabel": "R", "ylabel": "A", "title": "Axial"},
    }

    for i, axes_name in enumerate(axes_names):
        ax_map = axes_map[axes_name]
        axes[i].imshow(ax_map["slice"], aspect=ax_map["aspect"], **kwargs)
        if xlabels:
            axes[i].set_xlabel(ax_map["xlabel"])
        axes[i].set_ylabel(ax_map["ylabel"])
        axes[i].invert_xaxis()
        axes[i].set_title(ax_map["title"])

    if fig is not None:
        fig.tight_layout(pad=0)
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def plot_subject(
    subject: tio.Subject,
    cmap_dict=None,
    show=False,
    figsize=None,
    clear_axes=True,
    return_fig=False,
    single_axis=None,  # one of (Sagittal, Coronal, Axial)
    **kwargs,
):
    num_images = len(subject)
    many_images = num_images > 2
    subplots_kwargs = {"figsize": figsize}
    try:
        if clear_axes:
            subject.check_consistent_spatial_shape()
            subplots_kwargs["sharex"] = "row" if many_images else "col"
            subplots_kwargs["sharey"] = "row" if many_images else "col"
    except RuntimeError:  # different shapes in subject
        pass

    axes_names = "sagittal", "coronal", "axial"
    if single_axis is not None:
        assert single_axis in ("sagittal", "coronal", "axial")
        axes_names = (single_axis,)
    num_rows = len(axes_names)

    args = (num_rows, num_images) if many_images else (num_images, num_rows)
    fig, axes = plt.subplots(*args, **subplots_kwargs, squeeze=False)
    # The array of axes must be 2D so that it can be indexed correctly within
    # the plot_volume() function
    axes = axes.T if many_images else axes.reshape(-1, num_rows)
    iterable = enumerate(subject.get_images_dict(intensity_only=False).items())
    for image_index, (name, image) in iterable:
        image_axes = axes[image_index]
        cmap = None
        if cmap_dict is not None and name in cmap_dict:
            cmap = cmap_dict[name]
        last_row = image_index == len(axes) - 1
        plot_volume(
            image,
            axes=image_axes,
            show=False,
            cmap=cmap,
            xlabels=last_row,
            single_axis=single_axis,
            **kwargs,
        )
        for axis, axis_name in zip(image_axes, axes_names):
            axis.set_title(f"{name} ({axis_name})")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if show:
        plt.show()
    if return_fig:
        return fig
    return data
