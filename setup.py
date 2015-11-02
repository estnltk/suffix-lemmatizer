#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = 'suffix-lemmatizer',
    description = 'Suffix Lemmatizer for Estonian',
    version = '0.1',
    install_requires = ['docopt==0.6.2'],
    author = 'Alexander Tkachenko',
    packages = find_packages(),
    entry_points = {
        'console_scripts': [
          'train_suffixlemmatizer=suffix_lemmatizer.scripts.train:main',
          'test_suffixlemmatizer=suffix_lemmatizer.scripts.test:main',
         ]
    }
)

