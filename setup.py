from setuptools import setup, find_packages
import os
import sys

# Add src directory to path to import version and author
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from llm_trainer import __version__, __author__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements
requirements = [
    # Core ML libraries
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",
    
    # Hugging Face ecosystem
    "transformers>=4.30.0",
    "datasets>=2.12.0",
    "tokenizers>=0.13.0",
    "accelerate>=0.20.0",
    
    # Data processing
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    
    # Visualization and monitoring
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "tensorboard>=2.13.0",
    "wandb>=0.15.0",
    
    # Utilities
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "omegaconf>=2.3.0",
]

setup(
    name="llm-trainer",
    version=__version__,
    author=__author__,
    description="A complete framework for training Large Language Models from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "distributed": [
            "deepspeed>=0.9.0",
        ],
        "mixed-precision": [
            "apex",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-train=llm_trainer.scripts.train:main",
            "llm-generate=llm_trainer.scripts.generate:main",
            "llm-eval=llm_trainer.scripts.evaluate:main",
        ],
    },
)
