"""Setup file for neural network building blocks package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Regular requirements
requirements = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "pytest>=7.0.0",
    "typing-extensions>=4.0.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.5.0",
    "tensorboard>=2.12.0",
    "pyyaml>=6.0",
    "pytorch_basics_library>=0.1.0",
]

setup(
    name="nncore",
    version="0.1.0",
    author="Peyton Tolbert",
    author_email="email@peytontolbert.com",
    description="A comprehensive library of composable neural network components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peytontolbert/neural-network-building-blocks",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ]
    }
) 