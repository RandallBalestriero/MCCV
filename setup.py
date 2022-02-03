import setuptools
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MCCV",
    author="Randall Balestriero",
    author_email="rbalestriero@fb.com",
    description="Minimalist argument parser based cross-validation and cluster job submission file generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "fair_jumpstart"},
    # packages=setuptools.find_packages(where="fair_jumpstart"),
    packages=[
        "mccv",
    ],
    python_requires=">=3.6",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
