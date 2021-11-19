import random

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
