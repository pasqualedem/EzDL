import numpy as np
from matplotlib import colormaps


def tensor_to_segmentation_image(prediction, cmap: list = None, labels=None) -> np.array:
    if cmap is None:
        cmap = list(map(lambda t: tuple(map(lambda x: int(x * 256), t)), colormaps.get("Set1").colors))
    if labels is None:
        labels = np.unique(prediction)
    segmented_image = np.ones((*prediction.shape, 3), dtype="uint8")
    for i in range(len(labels)):
        segmented_image[prediction == i] = cmap[i]
    return segmented_image

