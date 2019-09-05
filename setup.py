import setuptools

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()
    
setuptools.setup(
    name = "ckiptagger",
    version = "0.0.16",
    author = "Peng-Hsuan Li",
    author_email = "jacobvsdanniel@gmail.com",
    description = "Neural implementation of CKIP WS, POS, NER tools",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/ckiplab/ckiptagger",
    python_requires = ">=3.6",
    packages = ["ckiptagger"],
    package_dir = {"ckiptagger": "src"},
    extras_require = {
        "tf": ["tensorflow"],
        "tfgpu": ["tensorflow-gpu"],
        "gdown": ["gdown"],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
