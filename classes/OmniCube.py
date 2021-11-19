from classes.OmniPix import OmniPix
from classes.OmniChunk import OmniChunk

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
