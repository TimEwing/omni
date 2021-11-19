### HERE BE DRAGONS
### This script generates images with all of the possible collors
### </dragons>

import re

from typing import Dict, List

import constants

def _printHelp():
    print("Usage: python omni FILENAME SIZE")

class _Args:
    def __init__():
        filename: str = constants._FILENAME
        size: int = constants._SIZE

def _parseArgs(argv: List[str]) -> _Args:
    args = _Args
    # Parse command line arguments
    if (len(argv) >= 3):
        # parse first arg
        # see if help or -h is typed
        if (not re.match(r'(-|--)(h)', argv[2]) == None):
            # Show help message
            _printHelp()

        args.filename = argv[1]
        if (len(argv) == 4):
            try:
                args.size = int(argv[1])
            except ValueError:
                raise Exception("SIZE arg (argv[2]) must be an integer")
    else:
        _printHelp()
        raise Exception("Incorrect number of arguments, see help above.")

    return args


if __name__ == '__main__':
    # import resource
    import sys

    import numpy as np
    import numpy.ma as ma

    # Custom Imports
    import constants
    from ImageTools import ImageTools
    from OmniCube import OmniCube

    # Parse command line arguments
    args = _parseArgs(sys.argv)
    constants._FILENAME = args.filename
    constants._SIZE = 128

    # resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
    # Let it be known that we may get segfaults if we recur too hard
    sys.setrecursionlimit(10**6)

    print("Opening image")
    input_image = ImageTools.get_image(constants._FILENAME, constants._SIZE)
    input_map = ImageTools.get_image(constants._FILENAME, constants._SIZE)
    print("  - Image shape:", input_image.shape)
    try:
        assert input_image.shape == input_map.shape
        # Remove alpha channel
        if (input_image.shape[2] == 4):
            input_image = input_image[:,:,:3]
        assert input_image.shape[2] == 3 
    except AssertionError:
        raise
    width, height, _ = input_image.shape

    print("Assigning chunks")
    input_map_reshaped = input_map.reshape(-1, input_map.shape[2]) # flat map
    chunks = np.unique(input_map_reshaped, axis=0, return_counts=True)
    chunks = list(zip(*chunks)) # make (color, count) tuples
    chunks = [(tuple(c), s) for c,s in chunks]
    print("  - Found {} chunks".format(len(chunks)))
    chunk_map = {}
    for x in range(width):
        for y in range(height):
            chunk_map[(x,y)] = tuple(input_map[x,y])

    # Build cube
    print("Building cube")
    cube = OmniCube(constants._SIZE, chunks)

    coords = [(x,y) for x in range(width) for y in range(height)]
    def key(coord):
        return (coord[0] - width/2) + (coord[1] - height/2)
    coords = sorted(coords, key=key)

    i = 0
    for x, y in coords:
        if i % 1000 == 0:
            print("\rAssigning new colors: {:.2f}%  ".format((i/len(coords)) * 100))
        i += 1

        target_color_list = []
        if x > 0:
            # if chunk_map[(x,y)] == chunk_map[(x-1,y)]:
            target_color_list.append(tuple(input_image[x-1,y]))
        if y > 0:
            # if chunk_map[(x,y)] == chunk_map[(x,y-1)]:
            target_color_list.append(tuple(input_image[x,y-1]))
        if x < width-2:
            # if chunk_map[(x,y)] == chunk_map[(x+1,y)]:
            target_color_list.append(tuple(input_image[x+1,y]))
        if y < height-2:
            # if chunk_map[(x,y)] == chunk_map[(x,y+1)]:
            target_color_list.append(tuple(input_image[x,y+1]))
        for _ in range(1):
            target_color_list.append(tuple(input_image[x,y]))
        target_color_channels = zip(*target_color_list)
        target_color = [sum(chans) // len(chans) for chans in target_color_channels]
        input_image[x,y] = cube.pop(chunk_map[(x,y)], target_color).color
    print("\nDone")
    # save output
    ImageTools.save_image('out.bmp', constants._SIZE, input_image)
