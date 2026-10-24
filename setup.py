from setuptools import setup, find_packages

setup(
    name="rl-adaptive-round-robin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29", "numpy>=1.26", "pandas>=2.0",
        "PyYAML>=6.0", "matplotlib>=3.8", "scipy>=1.11", "torch>=2.2"
    ],
)
