import pandas as pd
import matplotlib.pyplot as plt
from numpy import ceil


class DataVisualization(object):

    def multiple_plots(self, x_array, y_array, title_array, cols=2, figsize=(12, 8)):

        rows = int(ceil(len(x_array) / cols))
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

        i = 0
        for row in axes:
            for ax in row:
                x = x_array[i]
                y = y_array[i]
                ax.plot(x, y)
                ax.set_title(title_array[i])
                i += 1
        plt.tight_layout()
        plt.show()