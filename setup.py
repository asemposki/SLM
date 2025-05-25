from setuptools.command.install import install
import os
from setuptools import setup, find_packages


class CustomInstall(install):
    def run(self):
        os.system("sphinx-build -b html docs docs/_build/html")
        install.run(self)  # Ensure normal install process runs


setup(
    name="slm",
    version="0.1",
    description="Star Log-extended eMulator",
    authors=[
        "Sudhanva Lalit <lalit@frib.msu.edu>",
        "Alexandra Semposki <as727414@ohio.edu>",
        "Joshua Maldonado",
    ],
    package_dir={"": "version-0.1"},
    # packages=find_packages(where="version-0.2"),
    # include_package_data=True,
    # package_data={"slm": ["data/*"]},
    install_requires=["numpy", "scipy", "matplotlib", "pandas", "mkdocs"],
    # scripts=["install.sh"],
    # cmdclass={"install": CustomInstall},
)
