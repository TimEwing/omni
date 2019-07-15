
import math
import random
from heapq import heappush, heappop
from functools import total_ordering

_INF = float('infinity')

## Some style notes:
# rgb and xy tuples can be concat'd down, so (r, g, b) is (r,g,b) etc.
# line wrapping is at 100 chars (meaning only 99 chars per line)
@total_ordering
class OmniPix():
    def __init__(self, r, g, b):
        self.color = (r,g,b)
        self.potential = 0
        self.selected = False

    def __eq__(self, other):
        try:
            return self.color == other.color
        except AttributeError:
            return self.color == other

    def __gt__(self, other):
        # Compare channel 0, then 1, then 2
        if self.color[0] > other.color[0]:
            return True
        elif self.color[0] < other.color[0]:
            return False
        elif self.color[1] > other.color[1]:
            return True
        elif self.color[1] < other.color[1]:
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
        ## Easy sum of distances distance for now, maybe do euclidian later...
        # or maybe do hsv later? who the fuck knows
        return (
            abs(self.color[0] - other.color[0]) 
            + abs(self.color[1] - other.color[1])
            + abs(self.color[2] - other.color[2])
        )

class OmniCluster():
    def __init__(self, omni_pixels, chunk_pix, chunk_size=20, depth=0):
        self.omni_pixels = omni_pixels
        self.chunk_pix = chunk_pix
        self.chunk_size = chunk_size
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
        centroids = random.sample(self.omni_pixels, self.chunk_size)
        # Initialize centroid assignments
        centroid_assignments = {}
        for centroid in centroids:
            centroid_assignments[centroid] = []
        # Assign each pixel to a centroid
        for omni_pix in self.omni_pixels:
            centroid = min(centroids, key=omni_pix.dist)
            centroid_assignments[centroid].append(omni_pix)
        # Build tree
        self.tree = {}
        for centroid in centroid_assignments:
           self.tree[centroid] = OmniCluster(
            centroid_assignments[centroid], 
            self.chunk_pix,
            self.chunk_size,
            depth=self.depth+1,
        )

    def pop(self, target):
        if self.tree is None:
            # If this is the 'stem' (the last node from the bottom), just get the closest leaf
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

    def build(self, omni_cube, count):
        print("Building {}".format(self.chunk_pix.color))
        # Select the pixels that will go in this chunk
        selected_pix = []
        ## Use a set to track which pixels have been marked as adjacent and a heap to track their 
        # ordering by potential
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
                    # Push the pixel on to the heap as a (potential, pixel) tuple, subtracting off
                    # the potential from this chunk so we actually follow a reasonable contour
                    heappush(
                        adj_pixels_heap, 
                        (adj_pix.potential - (1 / adj_pix.dist(self.chunk_pix)), adj_pix)
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
                print('BAD ' + str(current_pix))
        print(len(selected_pix))
        # Construct a k-means cluster tree
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
                    # therefore cube[(r,g,b)] will always point to the right pixel; you can't have
                    # a duplicate in this cube (by definition)
                    new_pix = OmniPix(r, g, b)
                    self.cube[new_pix] = new_pix

        self.set_potentials()
        self.build_chunks()

    def set_potentials(self):
        for _, color in self.cube.items():
            potential = 0
            for chunk_pix in self.chunk_pixels:
                # potential is 1/x where x is the distance
                # gets set to inf at chunk colors so they should never get stolen by another chunk
                if color == chunk_pix:
                    potential = _INF
                    break
                potential += 1 / color.dist(chunk_pix)
            color.potential = potential

    def build_chunks(self):
        for chunk_pix in self.chunk_pixels:
            self.chunks[chunk_pix] = OmniChunk(chunk_pix)
        for chunk_color, chunk in self.chunks.items():
            chunk.build(self, self.chunk_sizes[chunk_color])

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

if __name__ == '__main__':
    depth = 128
    size = (depth**3) / 4
    # size = 20000
    chunks = [
        ((0, 0, 0), size),
        ((10, 0, 10), size),
        ((15, 0, 15), size),
        ((0, 15, 0), size),
    ]

    cube = OmniCube(depth, chunks)

    k = 0
    i = 0
    n = 0
    for r in range(depth):
        print("r =", r)
        for b in range(depth):
            for g in range(depth):
                n += 1
                k += 1
                # print(r,g,b, chunks[i][0], k)
                node = cube.pop(chunks[i][0], (r,g,b))
                if node is None:
                    i += 1
                    k = 0
                    cube.pop(chunks[i][0], (r,g,b))

