#%% Imports

from scipy import optimize
import numpy as np
import itertools

# %% Palette Class def
class Palette:

    def __init__(self, size, fixed = np.empty(0)):

        # Check to make sure we're making a big enough palette
        assert(size > fixed.shape[0]), "Size must be larger than fixed array"

        self.size = size
        self.fixed = fixed
        self.has_fixed = self.fixed.shape[0] > 0

        # Initial random colors in the palette to be optimized
        self.new_colors = np.random.rand(size - fixed.shape[0], 3)

    @staticmethod
    def color_distance(c1, c2):

        r_bar = (c1[0] + c2[0]) / 2

        d_squared = (c1 - c2) ** 2

        inside = ((2 + r_bar) * d_squared[0] +
                  4 * d_squared[1] +
                  (3 - r_bar) * d_squared[2])
        
        return inside ** 0.5


    # Objective function for optimizer
    def objective(self):
        
        if self.has_fixed:
            all_colors = np.concatenate(self.fixed, self.new_colors)
        else:
            all_colors = self.new_colors

        pairs = itertools.combinations(all_colors, 2)

        min_dist = (min(Palette.color_distance(x, y) for x, y in pairs))

        return min_dist


# %%
