from PIL import Image
import numpy as np

import constants
from ImageTools import ImageTools

size = constants._SIZE

img = np.zeros(shape=[size, size, 3])

# input_image = np.asarray(Image.open(constants._FILENAME))
# input_image = input_image[0:constants._SIZE,0:constants._SIZE]

ImageTools.save_image(filename=f"black_{size}.png", size=size, arr=img)