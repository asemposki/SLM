from setuptools.command.install import install
import os
from setuptools import setup, find_packages


setup(
    name="slm",
    version="0.1.0",
    description="Star Log-extended eMulator",
    authors=[
        "Sudhanva Lalit <lalit@frib.msu.edu>",
        "Alexandra Semposki <as727414@ohio.edu>",
        "Joshua Maldonado",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # include_package_data=True,
    # package_data={"slm": ["data/*"]},
    install_requires=["numpy", "scipy", "matplotlib", "pandas", "mkdocs"],
)
