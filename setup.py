from setuptools import setup, find_packages

setup(
    name="jaxl",
    description="Learning Codebase with JAX",
    version="0.1",
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.7",
        "optax>=0.1.4",
        "flax>=0.6.8",
        "black>=23.3.0",
        "tqdm>=4.65.0",
        "orbax-checkpoint>=0.1.8",
        "matplotlib>=3.7.1",
        "gymnasium>=0.28.1",
        "tensorboard>=2.13.0",
        "torch>=2.0.0",
        "sphinx>=7.0.1",
        "pyment>=0.3.3",
    ],
    extras_requires={},
    packages=find_packages(),
    include_package_data=True,
)
