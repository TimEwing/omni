from PIL import Image
import numpy as np

# Static Class
class ImageTools(object):
    def get_image(filename, size):
        input_image = np.asarray(Image.open(filename))
        input_image = input_image * (size/256)
        input_image = input_image.astype('uint8')
        return input_image

    def save_image(filename, size, arr):
        output_image = Image.fromarray((arr * (256/size)).astype('uint8'))
        output_image.save(filename)
        print("Saved to {}".format(filename))
