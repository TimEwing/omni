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

    def gen_random_grid(filename, gridsize, size):
        # output_image = np.zeros(shape=[gridsize, gridsize, 3], dtype=np.float32)
        # for x in range(gridsize):
        #     for y in range(gridsize):
        #         output_image[x,y] = np.random.ran * 256
        output_image = np.random.randn(gridsize**2).reshape([gridsize, gridsize])
        output_image = output_image.resize([size, size])
        ImageTools.save_image(filename, size, output_image)
