from setuptools import setup, find_packages

setup(
    name="my_project",
    version="0.1.0",
    packages=find_packages(where="library"),
    package_dir={"": "library"},
    install_requires=[
        "numpy",
        "scikit-learn",
        "scikit-image",
        "opencv-python",
        "matplotlib",
    ],
)
