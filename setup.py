import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml_tools",
    version="0.0.2",
    author="Tyler Renelle",
    author_email="tylerrenelle@gmail.com",
    description="Various ML tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lefnire/ml-tools",
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
