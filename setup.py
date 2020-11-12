import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Deep_Gene-yi_liu", # Replace with your own username
    version="0.1.1",
    author="Yi Liu",
    author_email="yil@uchicago.edu",
    description="Deep Gene Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires =[tensorflow>=2.2],
)