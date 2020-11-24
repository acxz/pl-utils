"""Setup file."""
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pl-utils',
    version='0.0.1',
    author='acxz',
    long_description=long_description,
    description='Some models and trainer scripts created using \
            pytorch-lightning.',
    packages=setuptools.find_packages(),
    install_requires=[
        'argparse',
        'pytorch-lightning',
        'torch',
    ],
)
