import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name = "ckiptagger",
    version = "0.0.11",
    author = "Peng-Hsuan Li",
    author_email = "jacobvsdanniel@gmail.com",
    description = "Neural implementation of CKIP WS, POS, NER tools",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/ckiplab/ckiptagger",
    packages = ["ckiptagger"],
    package_dir = {"ckiptagger": "src"},
    extras_require = {
        "tf": ["tensorflow"],
        "tfgpu": ["tensorflow-gpu"],
        "gdown": ["gdown"],
    },
    licence = "CC BY-NC-SA 4.0",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
    ],
)
