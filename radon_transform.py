import cv2
import numpy as np
from numpy import typing as npt

NDArray = npt.NDArray
MAX_N_ANGLES = 721


def radon(image: NDArray, n_angles: int):
    assert image.shape[0] == image.shape[1]
    assert n_angles > 0
    assert n_angles < MAX_N_ANGLES

    if image.ndim > 2:
        image = _bgr_to_grayscale(image)

    padded_image = _resize_and_pad(image)

    h, w = padded_image.shape[:2]
    cx, cy = w // 2, h // 2
    sinogram_image = np.zeros((n_angles, h))

    rhos = np.linspace(0, 180, n_angles, endpoint=False)
    for i, rho in enumerate(rhos):
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), rho, 1.0)
        rotated_image = cv2.warpAffine(padded_image, rotation_matrix, padded_image.shape[:2])
        sinogram_image[i, :] = rotated_image.sum(axis=0)

    return sinogram_image


def _resize_and_pad(image: NDArray):
    side = max(image.shape)
    image = cv2.resize(image, (side, side))
    diag = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
    pad_width = int(diag - image.shape[0]) // 2
    padded_image = np.pad(
        image,
        pad_width=[
            (pad_width, pad_width + 1 * (pad_width % 2)),
            (pad_width, pad_width + 1 * (pad_width % 2))
        ]
    )

    return padded_image


def _bgr_to_grayscale(image: NDArray) -> NDArray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
