__author__ = 'cvpia'

import numpy as np
import matplotlib.cm as cm
try:
    import PIL.Image as Image
except ImportError:
    import Image

def main():
    f = np.load('weigths_lasg0.npz')
    Image.fromarray(np.uint8(cm.gist_earth(np.rollaxis(f['arr_0'][0, :, :, :], 0, 2)))*255)

    Image.save('filters.png')

if __name__ == '__main__':
    main()