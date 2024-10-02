import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TSHIELD_isega24",
    version="0.0.1",
    author="Iván Sevillano Garcia",
    author_email="isevillano@ugr.es",
    description="Transformation-Selective Hidden Input Evaluation for Learning Dynamics",
    long_description=long_description,
    classifiers=["Programming Language :: Python :: 3"],
    packages=["TSHIELD", "TSHIELD.TSHIELD", "TSHIELD.procedures"],
    install_requires=[
        "torch",
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "numpy",
        "scipy",
        "pandas",
        "torchvision",
        "efficientnet_pytorch",
        "tqdm",
        "opencv-python",
        "IPython",
        "seaborn",
        "plotly",
        "ipywidgets",
        "sphinx",
        "sphinx_rtd_theme",
        "sphinxcontrib.bibtex",
        "nbsphinx",
        "wget",
        "pandoc",
        "ReVel",
        "torchsummary",
        "tensorboard",
    ],
)