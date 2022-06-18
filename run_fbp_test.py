import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from fbp import fbp


if __name__ == '__main__':
    base_path = os.path.abspath(os.path.dirname(__file__))
    outputs_dir = os.path.join(base_path, 'images', 'outputs')

    for angles_dir in sorted(os.listdir(outputs_dir), key=lambda key: int(key.split('_')[0])):
        output_dir = os.path.join(outputs_dir, angles_dir)

        sinogram = cv2.imread(os.path.join(output_dir, 'sinogram.jpg'))
        sinogram = cv2.cvtColor(sinogram, code=cv2.COLOR_BGR2GRAY)
        rhos = np.load(os.path.join(output_dir, 'angles.npy'))

        restored = fbp(sinogram, rhos)

        cv2.imwrite(os.path.join(output_dir, 'restored.jpg'),
                    np.clip(restored * 255, 0, 255).astype(np.uint8))

        plt.imshow(restored, cmap='Greys_r')
        plt.title(f'{" ".join(angles_dir.split("_"))}')
        plt.show()
