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
        "numpy==1.23",
        "opencv-python",
        "onnx==1.12.0",
        "plotly",
        "Pillow",
        "pyparsing==2.4.5",
        "ptflops",
        "ruamel.yaml",
        "scipy",
        "super-gradients==3.0.7",
        "torch==1.13.1+cu116@https://download.pytorch.org/whl/cu116",
        "torchvision==0.14.1+cu116@https://download.pytorch.org/whl/cu116",
        "torchaudio==0.13.1@https://download.pytorch.org/whl/cu116",
        "torchmetrics==0.8",
        "torchdistill",
        "tqdm",
        "wandb",
        "transformers",
        "einops==0.3.2",
        "scikit-learn",
        "streamlit",
        "streamlit-ace",
        "easydict",
        "clearml",
        "optuna"
    ]
)