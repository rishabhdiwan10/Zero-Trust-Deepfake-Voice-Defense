"""
Setup configuration for the Zero-Trust Deepfake Voice Defense System package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="zero-trust-deepfake-voice-defense",
    version="0.1.0",
    author="Rishabh Diwan",
    description=(
        "A multi-layered Zero-Trust Deepfake Voice Defense System combining "
        "CNN-based audio forensics, LangGraph agentic orchestration, and "
        "dynamic liveness verification."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rishabhdiwan10/Zero-Trust-Deepfake-Voice-Defense",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    entry_points={
        "console_scripts": [
            "ztdvd-train=scripts.train:main",
            "ztdvd-evaluate=scripts.evaluate:main",
            "ztdvd-benchmark=scripts.benchmark_latency:main",
        ],
    },
)
