#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


requirements = [
    "kipoi>=0.6.12",
    #$"git+ssh://git@github.com/DerThorsten/kipoi.git@niceparse",#egg=kipoi,
    "kipoi-utils>=0.1.12",
    # vep
    "pyvcf",
    "cyvcf2",
    "pybedtools",
    "pysam",  # required by pybedtools
    "intervaltree<3",
    "deepdish",
    "matplotlib",
    "seaborn",
    "shapely",
    "descartes",
    # TODO - which requirements to we need?
    "future",
    "numpy",
    "pandas",
    "tqdm",
    "related>=0.6.0",
    "enum34",
    "colorlog",
    "cookiecutter",
    # sometimes required
    "h5py",
    "urllib3>=1.21.1", #,<1.23",
]

test_requirements = [
    "bumpversion",
    "wheel",
    "jedi",
    "epc",
    "pytest>=3.3.1",
    "pytest-xdist",  # running tests in parallel
    "pytest-pep8",  # see https://github.com/kipoi/kipoi/issues/91
    "pytest-cov",
    "coveralls",
    "scikit-learn",
    "cython",
    # "genomelake",
    "keras",
    "tensorflow",
    "kipoiseq"
]

setup(
    name='kipoi_veff',
    version='0.3.0',
    description="kipoi_veff: variant effect prediction plugin for Kipoi",
    author="Kipoi team",
    author_email='avsec@in.tum.de',
    url='https://github.com/kipoi/kipoi-veff',
    long_description="kipoi_veff: variant effect prediction plugin for Kipoi",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": test_requirements,
    },
    # entry_points={'console_scripts': ['kipoi_veff = kipoi_veff.__main__:main']},
    license="MIT license",
    zip_safe=False,
    keywords=["variant effect prediction", "model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    include_package_data=True,
    tests_require=test_requirements
)
