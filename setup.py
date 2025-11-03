

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geospatial_tools",
    version="2.0.1",
    author="Giorgio Meschi",
    author_email="giorgio.meschi@cimafoundation.org",
    description="A pool of functions for geospatial analysis of raster and vector data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GiorgioMeschi/geospatial_tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'geopandas',
        'rasterio',
        'scipy',
        'toolz',
        'scipy',
        'contextily'
      ],
)


