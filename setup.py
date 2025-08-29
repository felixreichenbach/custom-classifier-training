from setuptools import find_packages, setup

setup(
    name="model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "keras==2.13.1rc0",
        "keras-cv==0.6.4",
        "Keras-Preprocessing==1.1.2",
        "tflite-support",
    ],
    include_package_data=True,
)
