import setuptools
import subprocess

# Use the first 7 digits of the git hash to set the version
version_root = "0.1"
try:
    __version__ = version_root+'.dev0+'+subprocess.check_output(['git', 'rev-parse', 'HEAD'])[:7].decode("utf-8")
except:
    __version__ = version_root

setuptools.setup(
    name="scarlet_test",
    packages=setuptools.find_packages(),
    version=__version__,
    description="Regression testing for the scarlet project",
    author="Fred Moolekamp, Peter Melchior, and Remy Joseph",
    author_email="peter.m.melchior@gmail.com",
    url="https://github.com/fred3m/scarlet_test",
    scripts=["bin/scarlet_test"]
)
