from setuptools import find_packages, setup

setup(
    name="pcasnn",
    version="1.0.0",
    description="PCA-guided neuron screening for Siamese networks under class imbalance",
    packages=find_packages(include=["pcasnn", "pcasnn.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<2.2",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "tensorflow>=2.15,<2.21",
        "scikit-posthocs>=0.9",
        "matplotlib>=3.7",
        "nltk>=3.8",
    ],
)
