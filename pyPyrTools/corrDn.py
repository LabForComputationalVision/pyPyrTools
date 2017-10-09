import ctypes

import numpy

# load the C library
from pyPyrTools import lib


def corrDn(image=None, filt=None, edges='reflect1', step=(1, 1),
           start=(0, 0), stop=None, result=None):
    if image is None or filt is None:
        print('Error: image and filter are required input parameters!')
        return
    else:
        image = image.copy()
        filt = filt.copy()

    if len(filt.shape) == 1:
        filt = numpy.reshape(filt, (1, len(filt)))

    if stop is None:
        stop = (image.shape[0], image.shape[1])

    if result is None:
        rxsz = len(list(range(start[0], stop[0], step[0])))
        rysz = len(list(range(start[1], stop[1], step[1])))
        result = numpy.zeros((rxsz, rysz))
    else:
        result = numpy.array(result.copy())

    if edges == 'circular':
        lib.internal_wrap_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 image.shape[1], image.shape[0],
                                 filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 filt.shape[1], filt.shape[0],
                                 start[1], step[1], stop[1], start[0], step[0],
                                 stop[0],
                                 result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    else:
        tmp = numpy.zeros((filt.shape[0], filt.shape[1]))
        lib.internal_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            image.shape[1], image.shape[0],
                            filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            filt.shape[1], filt.shape[0],
                            start[1], step[1], stop[1], start[0], step[0],
                            stop[0],
                            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            edges.encode('utf-8'))

    return result
