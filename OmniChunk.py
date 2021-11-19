import constants

from heapq import heappush, heappop

from OmniCluster import OmniCluster

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
                        [x for x in omni_cube.cube if x.potential < constants._INF and not x.selected], 
                        key=self.chunk_pix.dist,
                    )

        # Construct a k-means cluster tree
        print("      - Building tree", self.chunk_pix.color)
        self.tree = OmniCluster(selected_pix, self.chunk_pix)


    def pop(self, omni_pix):
        return self.tree.pop(omni_pix)
