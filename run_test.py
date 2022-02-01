import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from radon_transform import radon


if __name__ == '__main__':
    base_path = os.path.abspath(os.path.dirname(__file__))
    images_dir = os.path.join(base_path, 'images')

    image = cv2.imread(os.path.join(images_dir, 'input.jpg'))

    for n_angles in [30, 90, 180, 360]:
        sinogram = radon(image, n_angles)
        plt.imshow(sinogram, cmap='gray')
        plt.title(f'{n_angles} angles')
        plt.yticks(
            list(range(0, n_angles, 20)),
            labels=np.linspace(0, 180, n_angles, endpoint=False)[list(range(0, n_angles, 20))]
        )
        plt.show()

        sinogram = (sinogram / sinogram.max() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(images_dir, f'output_{n_angles}_angles.jpg'), sinogram)
