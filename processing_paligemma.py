from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def rescale(image: np.ndarray, scale: float, dtype:  np.dtype = np.float32) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def resize(image: Image, 
           size: Tuple[int, int], 
           resample: Image.Resampling = None,
           reducing_gap: Optional[int] = None
) -> np.ndarray:
    h, w = size
    resized_image = image.resize(
        (w, h), 
        resample=resample,
        reducing_gap=reducing_gap
        )
    return resized_image

def normalize(
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]]
    ) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image