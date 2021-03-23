from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='VQVAE-192',
    version='1.0',
    author='anonymous',
    description='VQVAE for the CLEVR-192 Dataset.',
    install_requires=requirements
)
