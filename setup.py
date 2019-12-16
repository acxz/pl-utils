import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nn-utils-pytorch",
    version="0.0.1",
    author="acxz",
    author_email="akashpatel2008@yahoo.com",
    description="A set of utility classes and functions for defining and\
    training pytorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/acxz/nn-sys-id",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
)
