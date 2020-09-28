import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lefnire_ml_utils",
    version="0.0.1",
    author="Tyler Renelle",
    author_email="tylerrenelle@gmail.com",
    description="Various ML utils.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lefnire/lefnire_ml_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
    ]
)
