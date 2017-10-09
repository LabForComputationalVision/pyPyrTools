from setuptools import (
    find_packages, Extension, setup
)

setup(
    name="pyPyrTools",
    version="1.0",
    url="https://github.com/LabForComputationalVision/pyPyrTools",
    description="Steerable pyramid library for image analysis",
    long_description="""pyPyrTools implements the steerable pyramid filter bank created by Simoncelli et alii in the 
    glorious 1990s. Python library adapted from the original C and Matlab code by Rob Young.""",
    license="MIT",
    packages=find_packages(exclude=['tests*']),
    install_requires=['numpy', 'matplotlib'],
    extras_require={
        'lxml': ['lxml'],
        'html5lib': ['html5lib'],
    },
    ext_modules=[Extension('pyPyrTools.wrapConv',
                           ['pyPyrTools/convolve.c', 'pyPyrTools/edges.c', 'pyPyrTools/wrap.c',
                            'pyPyrTools/internal_pointOp.c'])],

)
