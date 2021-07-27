
import math
import random
from heapq import heappush, heappop
from functools import total_ordering
import colorsys

from PIL import Image
import numpy as np

### HERE BE DRAGONS
# Let it be known that we may get segfaults if we recur too hard
import resource
import sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)
### </dragons>

_INF = float('infinity')
_SIZE = None

## Style note:
# line wrapping is at 100 chars (meaning only 99 chars per line) if I remember
@total_ordering
class OmniPix():
    def __init__(self, r, g, b):
        self.color = (r,g,b)
        self.potential = 0
        self.selected = False
        self.hsv = self._hsv()

    def __eq__(self, other):
        try:
            return self.color == other.color
        except AttributeError:
            return self.color == other

    def __gt__(self, other):
        if self.color[1] > other.color[1]:
            return True
        elif self.color[1] < other.color[1]:
            return False
        elif self.color[0] > other.color[0]:
            return True
        elif self.color[0] < other.color[0]:
            return False
        elif self.color[2] > other.color[2]:
            return True
        elif self.color[2] < other.color[2]:
            return False
        else:
            return False

    def __hash__(self):
        return hash(self.color)

    def __str__(self):
        return str(self.color) + ' ' + str(self.potential)

    def __repr__(self):
        return str(self.color) + ' ' + str(self.potential)

    def dist(self, other):
        # This function is a pretty abstract definition of 'color distance'
        # changing it pretty radically changes the output
        if self == other:
            return 0 # Don't change this; distance from this is always 0
        d = (
            # Hue is circular; take the shorter distance
            # The max difference in hue is half that of shade/value but it tends to be
            # even lower than that. For example, if shade is 50% for both, hue ends up
            # being 1/8 the weight of the other channels. Scaling by ~8ish gives good
            # results.
            min([
                (self.hsv[0] - other.hsv[0]) % _SIZE,
                (other.hsv[0] - self.hsv[0]) % _SIZE,
            ])
                * (self.hsv[1] / _SIZE) # Scale hue by normalized shade
                * (other.hsv[1] / _SIZE) # Repeat for the hue of the comparison color
                * 8
            + abs(self.hsv[1] - other.hsv[1]) # Add difference in shade
            + abs(self.hsv[2] - other.hsv[2]) # Add difference in value
        )
        return d

    # Separate function for setting potentials
    def get_potential(self, other):
        # Potentials used for slope-descent chunk assignment
        d = (
            min([
                (self.hsv[0] - other.hsv[0]) % _SIZE,
                (other.hsv[0] - self.hsv[0]) % _SIZE,
            ])
            + abs(self.hsv[1] - other.hsv[1])
            + abs(self.hsv[2] - other.hsv[2])
        )
        return 1/d if d != 0 else _INF

    def _hsv(self):
        # Get this color as HSV
        h,s,v = colorsys.rgb_to_hsv(
            self.color[0] / _SIZE, 
            self.color[1] / _SIZE, 
            self.color[2] / _SIZE, 
        )
        return (
            int(h * _SIZE),
            int(s * _SIZE),
            int(v * _SIZE),
        )

class OmniCluster():
    def __init__(self, omni_pixels, chunk_pix, chunk_size=4, depth=0):
        self.omni_pixels = omni_pixels
        self.chunk_pix = chunk_pix
        self.chunk_size = chunk_size # This is the number of leafs per stem
        self.tree = None
        self.leafs = None
        self.depth = depth

        self.build()

    def __str__(self):
        if self.tree is None:
            return ''

        as_str = ''
        for node in self.tree:
            as_str += ' ' + str(node) + ' ' + str(self.tree[node])
        return as_str

    def build(self):
        # If we don't have enough nodes left, setup leafs
        if len(self.omni_pixels) <= self.chunk_size:
            self.leafs = self.omni_pixels
            return
        # Pick centroids
        # This could be replaced with a better selection method for a more balanced tree
        # This method basically does random.sample(self.omni_pixels, self.chunk_size)
        # but pops the selected pixels from the array because efficiency
        # TODO: Rewrite this to actually be readable (sorry)
        if self.depth == 0:
            print("          - Picking centroids")
        centroids = [
            self.omni_pixels.pop(x) # Pop element x
            for x in sorted(random.sample(
                range(len(self.omni_pixels)), # Pick x from range(0, len(self.omni_pixels))
                self.chunk_size # pick self.chunk_size samples
            ))[::-1]
        ]

        # Initialize centroid assignments
        centroid_assignments = {}
        for centroid in centroids:
            centroid_assignments[centroid] = [centroid]
        # Assign each pixel to a centroid
        if self.depth == 0:
            print("          - Assinging pixels to centroids")
        for omni_pix in self.omni_pixels:
            centroid = min(centroids, key=omni_pix.dist)
            centroid_assignments[centroid].append(omni_pix)

        # Build tree
        self.tree = {}
        centroid_count = 0
        for centroid in centroid_assignments:
            if self.depth == 0:
                centroid_count += 1
                print(
                    "          - Building branch {} of {}  ".format(centroid_count,self.chunk_size),
                    end='\r'
                )
            try:
                self.tree[centroid] = OmniCluster(
                    centroid_assignments[centroid], 
                    self.chunk_pix,
                    self.chunk_size,
                    depth=self.depth+1,
                )
            except RecursionError:
                # It's bad news if we get here, but since we're doing random center selection
                # it can actually happen randomly. Let's hope it doesn't.
                print("Recursion Depth Exceeded:", self.depth + 1)
                exit()
        if self.depth == 0:
            print()

    def pop(self, target):
        # This is a (complicated) method for popping off the closest node to the target
        if self.tree is None:
            # If this node is a 'stem' (it has leafs, but those leaf nodes have no children), then
            # return the closest match among the leafs
            if len(self.leafs) == 0:
                return None
            node = min(self.leafs, key=lambda x: target.dist(x))
            self.leafs.remove(node)
            return node
        elif len(self.tree) == 0:
            return None
        else:
            while True:
                if len(self.tree) > 0:
                    # Get the result of a search of the closest node
                    subtree_color = min(self.tree, key=lambda x: target.dist(x))
                    node = self.tree[subtree_color].pop(target)
                    if node is None:
                        del self.tree[subtree_color]
                    else:
                        return node
                else:
                    return None

class OmniChunk():
    def __init__(self, chunk_pix):
        self.chunk_pix = chunk_pix
        self.tree = None

    def build(self, omni_cube, count, last=False):
        if last:
            # If there is only one chunk left, don't get fancy - just select everything not yet 
            # selected
            selected_pix = [x for x in omni_cube.cube if not x.selected]
        else:
            # Select the pixels that will go in this chunk
            selected_pix = []
            ## Use a set to track which pixels have been marked as adjacent and a heap to track 
            # their ordering by potential
            adj_pixels_set = set()
            adj_pixels_heap = []

            current_pix = omni_cube.cube[self.chunk_pix]
            while len(selected_pix) < count:
                # Add this pixel to selected 
                selected_pix.append(current_pix)
                current_pix.selected = True
                # Add adjacent pixels to heapq/set
                for adj_pix in omni_cube.adj(current_pix):
                    if adj_pix in adj_pixels_set:
                        pass
                    else:
                        adj_pixels_set.add(adj_pix)
                        # Push the pixel on to the heap as a (potential, pixel) tuple, subtracting
                        # off the potential from this chunk so we actually follow a reasonable 
                        # contour
                        heappush(
                            adj_pixels_heap, 
                            (adj_pix.potential - adj_pix.get_potential(self.chunk_pix), adj_pix)
                        )
                # Find next pixel
                if len(adj_pixels_set) > 0:
                    potential, current_pix = heappop(adj_pixels_heap)
                    # Remove selected pixel from set
                    adj_pixels_set.discard(current_pix)
                else:
                    # If we ran out of adjacents, pick the closest color and keep going unless we're 
                    # done filling this chunk
                    if len(selected_pix) == count:
                        break
                    current_pix = min(
                        [x for x in omni_cube.cube if x.potential < _INF and not x.selected], 
                        key=self.chunk_pix.dist,
                    )

        # Construct a k-means cluster tree
        print("      - Building tree", self.chunk_pix.color)
        self.tree = OmniCluster(selected_pix, self.chunk_pix)


    def pop(self, omni_pix):
        return self.tree.pop(omni_pix)

class OmniCube():
    def __init__(self, colorsize, chunks):
        # Chunks should be passed in as a list of ((r,g,b), size) tuples
        self.colorsize = colorsize
        self.colorsize_max = self.colorsize - 1 # goofy, but we only want to calculate it once
        self.chunk_pixels = [OmniPix(*color) for color, _ in chunks]
        self.chunk_sizes = {color: size for color, size in chunks}
        self.chunks = {}
        self.cube = {}
        # Init cube
        for r in range(colorsize):
            for g in range(colorsize):
                for b in range(colorsize):
                    ## Note that not all the data in new color is stored in the key; just the hash.
                    # but cube[(r,g,b)] will always point to the right pixel; you can't have
                    # a duplicate in this cube (by definition)
                    new_pix = OmniPix(r, g, b)
                    self.cube[new_pix] = new_pix

        self.set_potentials()
        self.build_chunks()

    def set_potentials(self):
        print("  - Setting potentials")
        for _, color in self.cube.items():
            potential = 0
            for chunk_pix in self.chunk_pixels:
                potential += color.get_potential(chunk_pix)
            color.potential = potential

    def build_chunks(self):
        for chunk_pix in self.chunk_pixels:
            self.chunks[chunk_pix] = OmniChunk(chunk_pix)
        for chunk_num, (chunk_color, chunk) in enumerate(self.chunks.items()):
            if chunk_num == len(self.chunk_pixels) - 1:
                last = True
            else:
                last = False
            print("  - Building chunk {}".format(chunk_color.color))
            chunk.build(self, self.chunk_sizes[chunk_color], last=last)

    def adj(self, color):
        r, g, b = color.color
        output_list = []
        if r > 0:
            output_list.append(self.cube[(r-1,g,b)])
        if r < self.colorsize_max:
            output_list.append(self.cube[(r+1,g,b)])
        if g > 0:
            output_list.append(self.cube[(r,g-1,b)])
        if g < self.colorsize_max:
            output_list.append(self.cube[(r,g+1,b)])
        if b > 0:
            output_list.append(self.cube[(r,g,b-1)])
        if b < self.colorsize_max:
            output_list.append(self.cube[(r,g,b+1)])
        output_list = [x for x in output_list if not self.cube[x].selected]
        return output_list

    def pop(self, chunk, target):
        chunk_pix = OmniPix(*chunk)
        target_pix = OmniPix(*target)
        return self.chunks[chunk_pix].pop(target_pix)

def get_image(filename, size):
    input_image = np.asarray(Image.open(filename))
    input_image = input_image * (size/256)
    input_image = input_image.astype('uint8')
    return input_image

def save_image(filename, size, arr):
    output_image = Image.fromarray((arr * (256/size)).astype('uint8'))
    output_image.save(filename)
    print("Saved to {}".format(filename))

if __name__ == '__main__':
    _SIZE = 128
    print("Opening image")
    input_image = get_image('black_128.png', _SIZE)
    input_map = get_image('black_128.png', _SIZE)
    print("  - Image shape:", input_image.shape)
    try:
        assert input_image.shape == input_map.shape
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
    cube = OmniCube(_SIZE, chunks)

    coords = [(x,y) for x in range(width) for y in range(height)]
    def key(coord):
        return (coord[0] - width/2) + (coord[1] - height/2)
    coords = sorted(coords, key=key)

    i = 0
    for x, y in coords:
        if i % 1000 == 0:
            print("Assigning new colors: {:.2f}%  ".format((i/len(coords)) * 100), end='\r')
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
    save_image('out.bmp', _SIZE, input_image)
