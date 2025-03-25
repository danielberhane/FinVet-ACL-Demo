from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="financial-misinfo",
    version="0.1.4",  # Updated to match your folder version
    description="Financial Misinformation Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Berhane Araya",
    author_email="dberhan4@gmu.edu",
    url="https://github.com/danielberhane/FinVet",  # Updated to your actual repo
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "financial_misinfo": ["data/*.bin", "data/*.pkl", "ui/*.py"],  # Added UI module
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20,<2.0",
        "faiss-cpu",  # Use faiss-gpu for GPU support
        "sentence-transformers",
        "requests",
        "tqdm",
        "pandas",
        "scikit-learn",
        "torch",
        "bert-score",
        "transformers",
        "asyncio",
        "nest-asyncio",
        "langchain",
        "langchain-community",
        "streamlit>=1.20.0",  # Added streamlit as a core dependency
    ],
    # Add extras_require for optional dependencies
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "ui": ["pyarrow==11.0.0"],  # Make pyarrow optional
    },
    entry_points={
        "console_scripts": [
            "financial-misinfo=financial_misinfo.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)