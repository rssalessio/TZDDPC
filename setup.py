from setuptools import setup, find_packages
from os import path


setup(name = 'TZDDPC',
    packages=find_packages(),
    version = '0.0.3',
    description = 'Python library Tube-Based Zonotopic Data-Driven Predictive Control',
    url = 'https://github.com/rssalessio/TZDDPC',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    install_requires=['numpy', 'scipy', 'cvxpy', 'dccp', 'pyzonotope', 'pydatadrivenreachability'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)