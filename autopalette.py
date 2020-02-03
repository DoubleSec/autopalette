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

    # Pure laziness, simply returns black and white in the correct format
    @staticmethod
    def black_white():

        return np.array([[0, 0, 0], [1, 1, 1]])

    def show_palette(self, cb_type = "", new_colors = None):

        if new_colors is None:
            new_colors = self.new_colors
        
        if cb_type != "":
            all_colors = self.as_cb(
                cb_type = cb_type, 
                new_colors = new_colors)
        
        else:
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
    
    # Basin-hopping optimizer
    def bh_optimize_palette(self, niter = 100, T = 1, stepsize = 0.5):
        
        res = optimize.basinhopping(
            func = self.objective,
            x0 = self.new_colors,
            niter = niter,
            T = T,
            stepsize = stepsize,
            minimizer_kwargs = {'method' : "L-BFGS-B",
                        'bounds' : [(0,1)] * self.new_colors.size}
        )
        
        res = np.clip(res.x, 0.0, 1.0)
        
        return np.reshape(res, (-1, 3))

    def __to_lms(self, new_colors):

        new_colors = np.reshape(new_colors, (-1, 3))

        rgb_to_lms = np.array(
            [[0.3140, 0.6395, 0.0465],
             [0.1554, 0.7579, 0.0867],
             [0.0178, 0.1094, 0.8726]]
        )

        lms_palette = np.transpose(new_colors)
        lms_palette = np.matmul(rgb_to_lms, lms_palette)
        lms_palette = np.transpose(lms_palette)

        return lms_palette

    # This method has no default values: it must take a palette as an input
    def __to_rgb(self, new_colors):

        new_colors = np.reshape(new_colors, (-1, 3))

        lms_to_rgb = np.array(
            [[5.4722, -4.6420, 0.1696],
             [-1.1252, 2.2931, -0.1679],
             [0.0298, -0.1932, 1.1636]]
        )

        rgb_palette = np.transpose(new_colors)
        rgb_palette = np.matmul(rgb_to_lms, rgb_palette)
        rgb_palette = np.transpose(rgb_palette)

        return rgb_palette

    # Protonopia is missing L cones
    def as_cb(self, cb_type = "protonopia", new_colors = None):

        if new_colors is None:
            new_colors = self.new_colors

        # Because apparently scipy doesn't keep the shape of stuff very well.
        new_colors = np.reshape(new_colors, (-1, 3))

        if self.has_fixed:
            all_colors = np.concatenate((self.fixed, new_colors))
        else:
            all_colors = new_colors

        lms_colors = self.__to_lms(all_colors)

        transform_dict = {
            "protonopia": np.array(
                [[0, 1.0512, -0.0512],
                [0, 1, 0],
                [0, 0, 1]]
            ),
            "deuteranopia": np.array(
                [[1, 0, 0],
                [0.9513, 0, 0.04867],
                [0, 0, 1]]
            ),
            "tritanopia": np.array(
                [[1, 0, 0],
                [0, 1, 0],
                [-0.8674, 1.8672, 0]]
            )
        }
        
        to_proto = transform_dict[cb_type]

        proto_pal = np.transpose(lms_colors)
        proto_pal = np.matmul(to_proto, proto_pal)
        proto_pal = np.transpose(proto_pal)

        return self.__to_rgb(proto_pal)

# %%
