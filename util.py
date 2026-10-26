import numpy as np
from PIL import Image


def tensor2im(image_tensor, imtype=np.uint8):
    """(C, H, W) tensor in [-1, 1] -> (H, W, 3) uint8 array."""
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    Image.fromarray(image_numpy).save(image_path)
