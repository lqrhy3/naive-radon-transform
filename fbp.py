import cv2
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq

from numpy import typing as npt
NDArray = npt.NDArray


def fbp(sinogram: NDArray, rhos: NDArray):
    assert len(sinogram.shape) == 2
    assert sinogram.shape[0] == len(rhos)

    n_angles, image_width = sinogram.shape

    restored = np.zeros((image_width, image_width))
    restored, pad_height, pad_width = _pad_to_square(restored)
    cx, cy = restored.shape[1] // 2, restored.shape[0] // 2

    sinogram = np.pad(
        sinogram,
        pad_width=[
            (0, 0),
            (pad_width, pad_width + 1 * (pad_width % 2))
        ]
    )

    rhos_consecutive = rhos[1:] - rhos[:-1]
    rhos_consecutive = np.insert(rhos_consecutive, 0, 0)

    i = 0
    for i, (signal, rho) in enumerate(zip(sinogram, rhos_consecutive)):
        f = fftfreq(sinogram.shape[1])
        fourier_filter = 2 * np.abs(f)
        filtered_fourie_signal1 = fft(signal, axis=0) * fourier_filter
        filtered_signal = np.real(ifft(filtered_fourie_signal1))

        signal_projected = np.tile(filtered_signal, (sinogram.shape[1], 1))
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), rho, 1.0)
        restored = cv2.warpAffine(restored, rotation_matrix, restored.shape[:2])
        restored += signal_projected

    restored = cv2.warpAffine(
        restored,
        cv2.getRotationMatrix2D((cx, cy), -rhos[i], 1.0),
        restored.shape[:2],
    )

    margin_top = int((image_width - image_width / np.sqrt(2)) / 2) + pad_height
    margin_bottom = -(int((image_width - image_width / np.sqrt(2)) / 2) + pad_height + 1 * (pad_height % 2))
    margin_left = int((image_width - image_width / np.sqrt(2)) / 2) + pad_width
    margin_right = -(int((image_width - image_width / np.sqrt(2)) / 2) + pad_width + 1 * (pad_width % 2))
    restored = restored[margin_top:margin_bottom, margin_left:margin_right]

    restored = (restored - restored.min()) / (restored.max() - restored.min())
    restored = np.clip(restored, 0, 1)
    return restored


def _pad_to_square(image: NDArray):
    diag = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
    pad_height = int(diag - image.shape[0]) // 2
    pad_width = int(diag - image.shape[1]) // 2
    padded_image = np.pad(
        image,
        pad_width=[
            (pad_height, pad_height + 1 * (pad_height % 2)),
            (pad_width, pad_width + 1 * (pad_width % 2))
        ]
    )

    return padded_image, pad_height, pad_width
