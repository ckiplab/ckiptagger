import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["ckiptagger"],
    package_dir={"ckiptagger": "src"},
)
