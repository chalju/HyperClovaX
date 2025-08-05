"""Setup script for HyperClovaX Python library."""

from setuptools import setup, find_packages

# Read requirements
with open("hyperclova/requirements.txt") as f:
    requirements = f.read().splitlines()

# Read the actual README.md file
with open("hyperclova/README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hyperclovax",
    version="0.1.0",
    description="Python library for HyperCLOVA X API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="chalju",
    author_email="chalju@example.com",
    url="https://github.com/chalju/HyperClovaX",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="hyperclova hyperclovax llm ai nlp embeddings chat naver clova",
)