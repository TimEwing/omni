import numpy as np

import constants
from ImageTools import ImageTools

size = constants._SIZE

img = np.zeros(shape=[size, size, 3])
ImageTools.save_image(filename=f"black_{size}.png", size=size, arr=img)