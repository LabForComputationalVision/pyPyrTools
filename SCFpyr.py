from .SFpyr import SFpyr
import numpy as np
from .rcosFn import rcosFn
from .pointOp import pointOp
import scipy
from .mkAngle import mkAngle


class SCFpyr(SFpyr):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, image, height, order, twidth, scale, n_scales, xp=np):    # (image, height, order, twidth, scale, n_scales)
        self.pyrType = 'steerableFrequency'
        self.image = image
        self.ht = height
        self.order = order
        self.twidth = twidth
        self.scale = scale
        self.n_scales = n_scales
        self.nbands = self.order+1

        self.xp = xp
        dims = np.array(self.image.shape)
        ctr = np.ceil((dims+0.5)/2).astype('int')

        (xramp, yramp) = self.xp.meshgrid((self.xp.arange(1, dims[1]+1)-ctr[1]) / (dims[1]/2.),
                                          (self.xp.arange(1, dims[0]+1)-ctr[0]) / (dims[0]/2.))
        angle = self.xp.arctan2(yramp, xramp)
        log_rad = self.xp.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        log_rad = self.xp.log2(log_rad)

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = rcosFn(self.twidth, (-self.twidth/2.0), np.array([0,1]))
        Yrcos = self.xp.sqrt(self.xp.array(Yrcos))

        YIrcos = self.xp.sqrt(1.0 - Yrcos**2)

        imdft = self.xp.fft.fftshift(self.xp.fft.fft2(self.image))
        if self.xp.__name__ == 'numpy':
            lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
            hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0], 0)
        else:
            lo0mask = self.xp.array(pointOp(self.xp.asnumpy(log_rad), self.xp.asnumpy(YIrcos), self.xp.asnumpy(Xrcos)[0],
                                            self.xp.asnumpy(Xrcos)[1] - self.xp.asnumpy(Xrcos)[0], 0))
            hi0mask = self.xp.array(pointOp(self.xp.asnumpy(log_rad), self.xp.asnumpy(Yrcos), self.xp.asnumpy(Xrcos)[0],
                                            self.xp.asnumpy(Xrcos)[1] - self.xp.asnumpy(Xrcos)[0], 0))
        self.pyr = []
        self.pyrSize = []

        hi0dft = imdft * hi0mask.reshape(imdft.shape[0], imdft.shape[1])
        hi0 = self.xp.fft.ifft2(self.xp.fft.ifftshift(hi0dft))

        self.pyr.append(self.xp.real(hi0.copy()))
        self.pyrSize.append(hi0.shape)

        lo0mask = lo0mask.reshape(imdft.shape[0], imdft.shape[1])
        lodft = imdft * lo0mask

        # self.pind = numpy.zeros((nbands, 2))
        self.bands = []
        for i in range(self.ht - 1, -1, -1):
            # Xrcos -= numpy.log2(2)
            Xrcos -= np.log2(1. / self.scale)

            lutsize = 1024
            Xcosn = np.pi * self.xp.array(self.xp.arange(-(2*lutsize+1), (lutsize+2))) / lutsize

            order = self.nbands - 1
            const = (2**(2*order))*(scipy.special.factorial(order, exact=True)**2) / \
                    float(self.nbands*scipy.special.factorial(2*order, exact=True))

            alfa = ( (np.pi+Xcosn) % (2.0*np.pi) ) - np.pi
            Ycosn = 2.0 * self.xp.sqrt(const) * (self.xp.cos(Xcosn)**order) * (self.xp.abs(alfa) < np.pi/2.0).astype(int)
            log_rad_tmp = self.xp.reshape(log_rad, (1,log_rad.shape[0]* log_rad.shape[1]))

            if self.xp.__name__ == 'numpy':
                himask = pointOp(log_rad_tmp, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0], 0)
            else:
                himask = self.xp.array(pointOp(self.xp.asnumpy(log_rad_tmp), self.xp.asnumpy(Yrcos), self.xp.asnumpy(Xrcos)[0],
                                               self.xp.asnumpy(Xrcos)[1] - self.xp.asnumpy(Xrcos)[0], 0))

            himask = himask.reshape(lodft.shape[0], lodft.shape[1])
            for b in range(self.nbands):
                angle_tmp = self.xp.reshape(angle, (1, angle.shape[0]*angle.shape[1]))

                if self.xp.__name__ == 'numpy':
                    anglemask = pointOp(angle_tmp, Ycosn, Xcosn[0] + np.pi*b/self.nbands, Xcosn[1] - Xcosn[0], 0)
                else:
                    anglemask = self.xp.array(pointOp(self.xp.asnumpy(angle_tmp), self.xp.asnumpy(Ycosn),
                                                      self.xp.asnumpy(Xcosn)[0] + np.pi * b / self.nbands,
                                                      self.xp.asnumpy(Xcosn)[1] - self.xp.asnumpy(Xcosn)[0], 0))
                anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
                banddft = (self.xp.sqrt(-1 + 0j)**order) * lodft * anglemask * himask
                band = self.xp.negative(self.xp.fft.ifft2(self.xp.fft.ifftshift(banddft)))
                self.bands.append(band[:].reshape(-1))
                self.pyr.append(band.copy())
                self.pyrSize.append(band.shape)
            dims = np.array(lodft.shape)
            ctr = np.ceil((dims+0.5)/2).astype('int')
            lodims = np.round(dims * self.scale ** (self.n_scales - self.ht + 1))
            loctr = np.ceil((lodims+0.5)/2).astype('int')
            lostart = ctr - loctr
            loend = (lostart + lodims).astype('int')

            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = self.xp.abs(self.xp.sqrt(1.0 - Yrcos**2))
            log_rad_tmp = self.xp.reshape(log_rad, (1,log_rad.shape[0]*log_rad.shape[1]))

            if self.xp.__name__ == 'numpy':
                lomask = pointOp(log_rad_tmp, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0],0)
            else:
                lomask = self.xp.array(pointOp(self.xp.asnumpy(log_rad_tmp), self.xp.asnumpy(YIrcos), self.xp.asnumpy(Xrcos)[0],
                                               self.xp.asnumpy(Xrcos)[1] - self.xp.asnumpy(Xrcos)[0], 0))
            lodft = lodft * lomask.reshape(lodft.shape[0], lodft.shape[1])

        self.bands = self.xp.concatenate(self.bands, 0)
        lodft = self.xp.fft.ifft2(self.xp.fft.ifftshift(lodft))
        self.pyr.append(self.xp.real(self.xp.array(lodft).copy()))
        self.pyrSize.append(lodft.shape)
        self.pyrSize = np.array(self.pyrSize)

    def band(self, bandNum):
        return self.xp.array(self.pyr[bandNum])

    def spyrHt(self):
        return self.ht

    def numBands(self):
        return self.order + 1

    def pyrLow(self):
        return self.xp.array(self.band(len(self.pyrSize)-1))

    def pyrHigh(self):
        return self.xp.array(self.band(0))

    # methods
    def reconPyr(self, levs='all', bands='all'):
        pind = self.pyrSize
        Nsc = int(np.round(np.log2(pind[0, 0] / pind[-1, 0].astype('float')) / np.log2(1 / self.scale)))
        Nor = (len(pind)-2) // Nsc

        pyrIdx = 1
        for nsc in range(Nsc):
            firstBnum = nsc * Nor + 1
            dims = pind[firstBnum][:]
            ctr = np.ceil((dims + 0.5) / 2).astype('int') #-1?
            ang = self.xp.array(mkAngle(dims, 0, ctr))
            ang[ctr[0]-1, ctr[1]-1] = -np.pi / 2.0
            for nor in range(Nor):
                nband = nsc * Nor + nor + 1
                ch = self.pyr[nband]
                ang0 = np.pi * nor / Nor
                xang = ((ang - ang0 + np.pi) % (2.0 * np.pi)) - np.pi
                amask = 2 * (self.xp.abs(xang) < (np.pi/2.0)).astype(int) + (self.xp.abs(xang) == (np.pi/2.0)).astype(int)
                amask[ctr[0]-1, ctr[1]-1] = 1
                amask[:,0] = 1
                amask[0,:] = 1
                amask = self.xp.fft.fftshift(amask)
                ch = self.xp.fft.ifft2(amask * self.xp.fft.fft2(ch))  # 'Analytic' version
                # f = 1.000008  # With this factor the reconstruction SNR
                                # goes up around 6 dB!
                f = 1
                ch = f*0.5*self.xp.real(ch)   # real part
                self.pyr[pyrIdx] = ch
                pyrIdx += 1

        res = self._reconSFpyr(levs, bands)

        return res

    def _reconSFpyr(self, levs='all', bands='all'):
        nbands = self.numBands()

        maxLev = 1 + self.spyrHt()
        if isinstance(levs, str) and levs == 'all':
            levs = np.array(list(range(maxLev + 1)))
        elif isinstance(levs, str):
            raise ValueError("Error: %s not valid for levs parameter. "
                             "levs must be either a 1D numpy array or the string 'all'." % (levs))
        else:
            levs = np.array(levs)

        if isinstance(bands, str) and bands == 'all':
            bands = np.array(list(range(nbands)))
        elif isinstance(bands, str):
            raise ValueError("Error: %s not valid for bands parameter. "
                             "bands must be either a 1D numpy array or the string 'all'." % (bands))
        else:
            bands = np.array(bands)

        # -------------------------------------------------------------------
        # matlab code starts here
        pind = self.pyrSize
        dims = np.array(self.pyrSize[0])
        ctr = np.ceil((dims + 0.5) / 2.0).astype('int')

        (xramp, yramp) = self.xp.meshgrid((self.xp.arange(1, dims[1] + 1) - ctr[1]) / (dims[1] / 2.),
                                        (self.xp.arange(1, dims[0] + 1) - ctr[0]) / (dims[0] / 2.))
        angle = self.xp.arctan2(yramp, xramp)
        log_rad = self.xp.sqrt(xramp ** 2 + yramp ** 2)
        log_rad[ctr[0] - 1, ctr[1] - 1] = log_rad[ctr[0] - 1, ctr[1] - 2]
        log_rad = self.xp.log2(log_rad)

        ## Radial transition function (a raised cosine in log-frequency):
        (Xrcos, Yrcos) = rcosFn(self.twidth, (-self.twidth / 2.0), np.array([0, 1]))
        Xrcos = self.xp.array(Xrcos)
        Yrcos = self.xp.sqrt(self.xp.array(Yrcos))
        YIrcos = self.xp.sqrt(self.xp.abs(1.0 - Yrcos ** 2))

        resdft = self._reconvSFpyr(self.pyr[1:], pind[1:], log_rad, Xrcos, Yrcos, angle, nbands, levs, bands, self.scale)

        # apply lo0mask
        if self.xp.__name__ == 'numpy':
            lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0], 0)
        else:
            lo0mask = self.xp.array(pointOp(self.xp.asnumpy(log_rad), self.xp.asnumpy(YIrcos), self.xp.asnumpy(Xrcos)[0],
                                            self.xp.asnumpy(Xrcos)[1] - self.xp.asnumpy(Xrcos)[0], 0))
        resdft = resdft * lo0mask

        # residual highpass subband
        if self.xp.__name__ == 'numpy':
            hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0], 0)
        else:
            hi0mask = self.xp.array(pointOp(self.xp.asnumpy(log_rad), self.xp.asnumpy(Yrcos), self.xp.asnumpy(Xrcos)[0],
                                            self.xp.asnumpy(Xrcos)[1] - self.xp.asnumpy(Xrcos)[0], 0))

        hi0mask = hi0mask.reshape(resdft.shape[0], resdft.shape[1])
        if 0 in levs:
            resdft += self.xp.fft.fftshift(self.xp.fft.fft2(self.pyr[0])) * hi0mask

        outresdft = self.xp.real(self.xp.fft.ifft2(self.xp.fft.ifftshift(resdft)))
        return outresdft

    def _reconvSFpyr(self, pyr, pind, log_rad, Xrcos, Yrcos, angle, nbands, levs, bands, scale):
        lo_ind = nbands + 1
        dims = pind[0]
        ctr = np.ceil((dims + 0.5) / 2).astype('int')
        XXrcos = self.xp.copy(Xrcos - self.xp.log2(1. / scale))

        if (levs > 1).any():
            lodims = pind[bands[-1] + 1]
            loctr = np.ceil((lodims + 0.5) / 2).astype('int')
            lostart = ctr - loctr + 1
            loend = lostart + lodims - 1
            nlog_rad = self.xp.copy(log_rad[lostart[0] - 1:loend[0], lostart[1] - 1:loend[1]])
            nangle = self.xp.copy(angle[lostart[0] - 1: loend[0], lostart[1] - 1: loend[1]])

            if pind.shape[0] > lo_ind:
                nresdft = self._reconvSFpyr(pyr[lo_ind - 1:len(pyr)], pind[lo_ind - 1:], nlog_rad, XXrcos, Yrcos,
                                            nangle, nbands, levs - 1, bands, scale)
            else:
                nresdft = self.xp.fft.fftshift(self.xp.fft.fft2(pyr[lo_ind - 1]))

            YIrcos = self.xp.sqrt(self.xp.abs(1. - Yrcos ** 2))

            if self.xp.__name__ == 'numpy':
                lomask = pointOp(nlog_rad, YIrcos, XXrcos[0], XXrcos[1] - XXrcos[0], 0)
            else:
                lomask = self.xp.array(pointOp(self.xp.asnumpy(nlog_rad), self.xp.asnumpy(YIrcos), self.xp.asnumpy(XXrcos)[0],
                                               self.xp.asnumpy(XXrcos)[1] - self.xp.asnumpy(XXrcos)[0], 0))

            resdft = self.xp.zeros(dims).astype('complex128')
            resdft[lostart[0] - 1:loend[0], lostart[1] - 1:loend[1]] = nresdft * lomask
        else:
            resdft = self.xp.zeros(dims)

        if (levs == 1).any():
            lutsize = 1024
            Xcosn = np.pi * self.xp.arange(-(2*lutsize + 1), lutsize + 1) / lutsize
            order = nbands - 1
            const = 2 ** (2*order) * (scipy.special.factorial(order) ** 2) / (nbands * scipy.special.factorial(2.*order))
            Ycosn = self.xp.sqrt(const) * (self.xp.cos(Xcosn)) ** order

            if self.xp.__name__ == 'numpy':
                himask = pointOp(log_rad, Yrcos, XXrcos[0], XXrcos[1] - XXrcos[0], 0)
            else:
                himask = self.xp.array(pointOp(self.xp.asnumpy(log_rad), self.xp.asnumpy(Yrcos), self.xp.asnumpy(XXrcos)[0],
                                               self.xp.asnumpy(XXrcos)[1] - self.xp.asnumpy(XXrcos)[0], 0))

            ind = 0
            for b in range(nbands):
                if (bands == b).any():
                    if self.xp.__name__ == 'numpy':
                        anglemask = pointOp(angle, Ycosn, Xcosn[0] + np.pi*b/nbands, Xcosn[1] - Xcosn[0], 0)
                    else:
                        anglemask = self.xp.asarray(pointOp(self.xp.asnumpy(angle), self.xp.asnumpy(Ycosn), self.xp.asnumpy(Xcosn)[0] +
                                                       np.pi * b / nbands, self.xp.asnumpy(Xcosn)[1] - self.xp.asnumpy(Xcosn)[0], 0))
                    band = pyr[ind]
                    banddft = self.xp.fft.fftshift(self.xp.fft.fft2(band))
                    resdft += (self.xp.power(-1 + 0j, 0.5)) ** (nbands - 1) * banddft * anglemask * himask
                ind += 1
        return resdft
