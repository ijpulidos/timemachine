"""setuptools based setup module.

Adapted from https://github.com/pypa/sampleproject/blob/main/setup.py
"""

import multiprocessing
import os
import pathlib
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

import versioneer


# CMake configuration adapted from https://github.com/pybind/cmake_example
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{multiprocessing.cpu_count()}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

cmdclass = versioneer.get_cmdclass()
cmdclass.update(build_ext=CMakeBuild)

setup(
    name="timemachine",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    description="A high-performance differentiable molecular dynamics, docking and optimization engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/proteneer/timemachine",
    author="Relay Therapeutics",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    keywords="molecular dynamics",
    ext_modules=[CMakeExtension("timemachine.lib.custom_ops", "timemachine/cpp")]
    if not os.environ.get("SKIP_CUSTOM_OPS")
    else [],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "grpcio",
        "importlib-resources",
        "jax",
        "jaxlib>0.3.13",
        "networkx",
        "numpy",
        "pymbar>3.0.4",
        "pyyaml",
        "scipy",
        "typing-extensions",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "black==21.10b0",
            "click==8.0.4",  # pinned due to https://github.com/psf/black/issues/2964 -- unpin if upgrading black
            "flake8==4.0.1",
            "grpcio-tools==1.30.0",
            "isort==5.10.1",
            "mypy==0.942",
            "pre-commit==2.17.0",
        ],
        "test": ["pytest", "pytest-cov", "hilbertcurve==1.0.5"],
    },
    package_data={
        "timemachine": [
            "datasets/freesolv/freesolv.sdf",
            "testsystems/data/5dfr_solv_equil.pdb",
            "testsystems/data/ligands_40.sdf",
            "testsystems/data/mobley_820789.sdf",
            "testsystems/data/hif2a_nowater_min.pdb",
            # NOTE: C++ sources used at runtime for JIT compilation
            "cpp/src/*.hpp",
            "cpp/src/kernels/*.cuh",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "sample=sample:main",
    #     ],
    # },
    project_urls={
        "Bug Reports": "https://github.com/proteneer/timemachine/issues",
        "Source": "https://github.com/proteneer/timemachine/",
    },
)
