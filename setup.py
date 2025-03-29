#setting up path

from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='RG-Attn',
    version=__version__,
    packages=find_packages(),
    license='cc-by-nc-4.0',
    author='Lantao Li',
    author_email='1152571959@qq.com',
    description='Radian Glue Attention with its two deployment cooperative perception frameworks',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
