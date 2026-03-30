"""Setup configuration for Document Classification OpenEnv"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="openenv-doc-classifier",
    version="1.0.0",
    author="OpenEnv Community",
    description="Document Classification and Routing OpenEnv Environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openenv/doc-classifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "openenv-doc-classifier=baseline_inference:run_baseline_evaluation",
        ],
    },
)
