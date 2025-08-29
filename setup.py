from setuptools import find_packages, setup

setup(
    name="model",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow==2.14.1",
        "tflite-support",
    ],
)
