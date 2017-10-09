import scipy.misc
from pylab import *

from pyPyrTools import Spyr

# import image to a numpy array and display
# im = array(scipy.misc.imresize(imread('buska.png'), 0.5), dtype=float64)
# im = array(imread('buska.png'), dtype=float64)

# im = array(imread('pizza.png')[:,:,0], dtype=float64)
im = array(scipy.misc.imresize(imread('pizza.png')[:, :, 0], 0.25), dtype=float64)

ion()
imshow(im)

pyr = Spyr(im, 3, 'sp3Filters')

figure()
subplot(1, 2, 1)
imshow(im)
subplot(1, 2, 2)
pyr.showPyr()
tight_layout()

# figure()
# for j in range(3):
#     for k in range(3):
#         subplot(3,3,j*3+k+1)
#         imshow(pyr.band(j*3+k))
