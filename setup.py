from setuptools import setup, find_packages

setup(
    name="jaxl",
    description="Learning Codebase with JAX",
    version="0.1",
    python_requires=">=3.9",
    install_requires=[],
    extras_requires={},
    packages=find_packages(include=["jaxl"]),
    include_package_data=True,
)
