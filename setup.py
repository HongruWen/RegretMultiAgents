from setuptools import setup, find_packages

setup(
    name="regrets",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "stable-baselines3",
        "pettingzoo",
    ],
) 