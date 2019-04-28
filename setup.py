import setuptools


setuptools.setup(
    name="tf-ddpg",
    version=0.1,
    author="Evan Hubinger",
    install_requires=[
        "coconut",
        "tensorflow-gpu",
    ],
    packages=setuptools.find_packages(),
)
