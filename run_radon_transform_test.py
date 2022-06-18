import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from radon_transform import radon


if __name__ == '__main__':
    base_path = os.path.abspath(os.path.dirname(__file__))
    inputs_dir = os.path.join(base_path, 'images', 'inputs')
    outputs_dir = os.path.join(base_path, 'images', 'outputs')

    image = cv2.imread(os.path.join(inputs_dir, 'input.jpg'))

    for n_angles in [30, 90, 180, 360]:
        rhos = np.linspace(0, 180, n_angles, endpoint=False)
        sinogram = radon(image, rhos)

        plt.imshow(sinogram, cmap='gray')
        plt.title(f'{n_angles} angles')
        plt.yticks(
            list(range(0, n_angles, 20)),
            labels=np.linspace(0, 180, n_angles, endpoint=False)[list(range(0, n_angles, 20))]
        )
        plt.show()

        sinogram = (sinogram / sinogram.max() * 255).astype(np.uint8)

        output_dir = os.path.join(outputs_dir, f'{n_angles}_angles')
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'sinogram.jpg'), sinogram)
        np.save(os.path.join(output_dir, 'angles.npy'), rhos)
