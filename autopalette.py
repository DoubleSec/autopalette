#%% Imports

from scipy import optimize
import numpy as np
import itertools
import matplotlib.pyplot as plt

# %% Palette Class def
class Autopalette:

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

    # Pure laziness, simply returns black and white in the correct format
    @staticmethod
    def black_white():

        return np.array([[0, 0, 0], [1, 1, 1]])


    # Objective function for optimizer
    def objective(self, new_colors = None):
        
        if new_colors is None:
            new_colors = self.new_colors

        # Because apparently scipy doesn't keep the shape of stuff very well.
        new_colors = np.reshape(new_colors, (-1, 3))

        if self.has_fixed:
            all_colors = np.concatenate((self.fixed, new_colors))
        else:
            all_colors = new_colors

        pairs = itertools.combinations(all_colors, 2)

        min_dist = min(Autopalette.color_distance(x, y) for x, y in pairs)

        return -min_dist

    def show_palette(self, new_colors = None):

        if new_colors is None:
            new_colors = self.new_colors
        
        if self.has_fixed:
            all_colors = np.concatenate((self.fixed, new_colors))
        else:
            all_colors = new_colors

        return plt.imshow([all_colors])

    # Simple optimizer
    def simple_optimize_palette(self):

        res =  optimize.minimize(
            fun = self.objective,
            method = "L-BFGS-B",
            x0 = self.new_colors,
            bounds = [(0,1)] * self.new_colors.size
        )

        return np.reshape(res.x, (-1, 3))


# %%
