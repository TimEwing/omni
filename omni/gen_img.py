from PIL import Image
import numpy as np

import constants
from ImageTools import ImageTools

size = constants._SIZE, constants._SIZE

img = np.zeros(shape=[size[0], size[0], 3])

# img = Image.open(constants._FILENAME)
# img_arr = np.asarray(Image.open(constants._FILENAME))

# img_arr = img_arr[0:constants._SIZE,0:constants._SIZE,:3]
# img.thumbnail(size)
# img = img.crop(box=(0, 0, size[0], size[1]))

# out_img = Image.fromarray(img)
# out_img.save(f"img_{size}.png", "PNG")

ImageTools.save_image(filename=f"black_{size}.png", size=size[0], arr=img)

# ImageTools.gen_random_grid("randomgrid_16.png", 16, 512)