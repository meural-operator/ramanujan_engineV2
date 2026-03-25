from setuptools import setup, find_packages

setup(
    name="ramanujan_home",
    version="1.0.0",
    description="Ramanujan@Home: Universal Distributed Scientific Computing Framework",
    packages=find_packages(),
    install_requires=[
        "mpmath==1.3.0",
        "numpy==2.3.5",
        "scipy==1.17.1", 
        "sympy==1.14.0",
        "tqdm",
        "requests",
        "pyrebase4",
        "pybloom_live",
        "ortools"
    ],
)
