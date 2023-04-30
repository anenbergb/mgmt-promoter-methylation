from setuptools import setup, find_packages

setup(
    name="mgmt",
    version="1.0.0",
    url="https://github.com:anenbergb/mgmt-promoter-methylation.git",
    author="Bryan Anenberg",
    author_email="anenbergb@gmail.com",
    description="Python libraries developed for the MGMT Promoter Methylation Classification project.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "monai",
        "tensorboard",
        "tensorboard-plugin-3d",
        "pandas",
        "pandas-stubs",
        "opencv-python",
        "loguru",
        "matplotlib",
        "ffmpeg-python",
        "tqdm",
        "types-tqdm",
    ],
    extras_require={
        "notebook": [
            "jupyter",
            "itkwidgets",
            "jupyter_contrib_nbextensions",
            "plotly",
        ],
        "dev": ["black", "mypy", "flake8"],
    },
)
