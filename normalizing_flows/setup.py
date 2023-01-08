from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = "0.0.1"

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs]

setup(
    name="normflows",
    version=__version__,
    description="Pytorch implementation of normalizing flows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haruishi43/normalizing-flows",
    download_url="",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    keywords="",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    author="",
    author_email="",
    install_requires=install_requires,
)
