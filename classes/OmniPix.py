from functools import total_ordering
import constants

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
                (self.hsv[0] - other.hsv[0]) % constants._SIZE,
                (other.hsv[0] - self.hsv[0]) % constants._SIZE,
            ])
                * (self.hsv[1] / constants._SIZE) # Scale hue by normalized shade
                * (other.hsv[1] / constants._SIZE) # Repeat for the hue of the comparison color
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
                (self.hsv[0] - other.hsv[0]) % constants._SIZE,
                (other.hsv[0] - self.hsv[0]) % constants._SIZE,
            ])
            + abs(self.hsv[1] - other.hsv[1])
            + abs(self.hsv[2] - other.hsv[2])
        )
        return 1/d if d != 0 else constants._INF

    def _hsv(self):
        # Get this color as HSV
        h,s,v = colorsys.rgb_to_hsv(
            self.color[0] / constants._SIZE, 
            self.color[1] / constants._SIZE, 
            self.color[2] / constants._SIZE, 
        )
        return (
            int(h * constants._SIZE),
            int(s * constants._SIZE),
            int(v * constants._SIZE),
        )
