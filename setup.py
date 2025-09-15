#!/usr/bin/env python
"""
B-Confident: Perplexity-Based Adjacency for Uncertainty Quantification in LLMs

A Python SDK implementing the PBA methodology for enterprise-grade uncertainty
quantification in Large Language Models with regulatory compliance support.
"""

from setuptools import setup, find_packages
import os

def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from README
long_description = ""
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="b-confident",
    version="0.1.0",
    author="Javier Marin",
    author_email="javier@jmarin.info",
    description="Perplexity-Based Adjacency for Uncertainty Quantification in Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/javiermarin/B-confident",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt") if os.path.exists("requirements.txt") else [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "serving": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "torchserve>=0.7.0",
            "ray[serve]>=2.0.0",
        ],
        "monitoring": [
            "mlflow>=2.0.0",
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "torchserve>=0.7.0",
            "ray[serve]>=2.0.0",
            "mlflow>=2.0.0",
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "b-confident=b_confident.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "b_confident": [
            "compliance/templates/*.json",
            "compliance/templates/*.md",
            "config/*.yaml",
        ],
    },
    zip_safe=False,
)