from setuptools import setup, find_packages
from os import path


setup(name = 'SZDPC',
    packages=find_packages(),
    version = '0.0.2',
    description = 'Python library Scenario-Based Zonotopic Data-Driven Predictive Control',
    url = 'https://github.com/rssalessio/SZDPC',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    install_requires=['numpy', 'scipy', 'cvxpy'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)