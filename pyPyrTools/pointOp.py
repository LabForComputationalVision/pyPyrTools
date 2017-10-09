import ctypes

import numpy

# load the C library
from pyPyrTools import lib


def pointOp(image, lut, origin, increment, warnings):
    result = numpy.zeros((image.shape[0], image.shape[1]))

    lib.internal_pointop(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         image.shape[0] * image.shape[1],
                         lut.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         lut.shape[0],
                         ctypes.c_double(origin),
                         ctypes.c_double(increment), warnings)

    return numpy.array(result)
