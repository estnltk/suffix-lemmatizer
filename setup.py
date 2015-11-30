#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = 'suffix-lemmatizer',
    description = 'Suffix Lemmatizer for Estonian',
    version = '1.0',
    install_requires = ['docopt==0.6.2'],
    author = 'Alexander Tkachenko',
    packages = find_packages(),
    package_data = {'suffix_lemmatizer': ['data/*.bz2']},
)

