import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ezdl",
    version="0.0.7",
    author="Pasquale De Marinis",
    author_email="pas.demarinis@gmail.com",
    url="https://github.com/pasqualedem/EzDL",
    description="A Simple Tool to make Deep Learning projects easier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.7',                # Minimum version requirement of the package
    entry_points={
        "console_scripts": [
            "ezdl=ezdl.cli:main",
        ]
    },
    install_requires=[
        "adjectiveanimalnumber",
        "click",
        "codecarbon",
        "matplotlib",
        "numpy",
        "opencv-python",
        "plotly",
        "Pillow",
        "ptflops",
        "ruamel.yaml",
        "scipy",
        "super-gradients==3.0.7",
        "torch",
        "torchaudio",
        "torchmetrics",
        "torchvision",
        "torchdistill",
        "tqdm",
        "wandb",
        "transformers",
        "einops",
        "streamlit",
        "streamlit-ace",
        "easydict",
        "clearml"
    ]
)