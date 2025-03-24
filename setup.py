from setuptools import setup, find_packages

setup(
    name="supply-chain-forecaster",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Read from requirements.txt
        line.strip()
        for line in open("requirements.txt")
        if not line.startswith("#") and line.strip()
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade supply chain forecasting system with advanced ML models",
    keywords="supply-chain, forecasting, machine-learning",
    url="https://github.com/yourusername/supply-chain-forecaster",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
)