import matplotlib.pyplot as plt
import numpy as np

def filter_slideshow(pdf_series)
    
    A = 'A_'
    keys = len(pdf_series.keys())
    for i in range(keys):
        if i == 0:
            p = plt.imshow(z)
            fig = plt.gcf()
            plt.clim()   # clamp the color limits
            plt.title("Boring slide show")
        else:
            z = z + 2
            p.set_data(z)

        print("step", i)
        plt.pause(0.5)
