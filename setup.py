
import os
from setuptools import find_packages, setup

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="swag_transformers",
    description="SWAG transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sami Virpioja, Hande Celikkanat",
    author_email="sami.virpioja@helsinki.fi",
    url="https://github.com/Helsinki-NLP/swag_transformers",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers>=4.30",
        "swa_gaussian>=0.1.2"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
    use_scm_version=True
)
