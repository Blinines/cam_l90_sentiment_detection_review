# -*- coding: utf-8 -*-
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='cam_l90_sentiment_detection_review',
    version='1.0',
    packages=['helpers', 'part_i_naive_bayes', 'part_ii_svm', 'private', 
              'ressources', 'data_model', 'tsne_python'],
    url='',
    license='',
    author='ines_blin',
    author_email='ines.blin@student.ecp',
    description='',
    install_requires=requirements,
)