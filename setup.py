from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="satprocess",
    version="0.1.0",
    author="Finlay Davis",
    author_email="finlayjdavis@gmail.com",
    description="Astronomical image calibration pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FinlayDavis/satprocess",
    packages=["satprocessing"],
    python_requires=">=3.7",
    license="BSD 3-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)