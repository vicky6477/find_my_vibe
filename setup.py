# setup.py
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------- #
#  Read long-description & install-requires directly from files
# ---------------------------------------------------------------------- #
readme = (ROOT / "README.md").read_text(encoding="utf-8")

requirements = (
    (ROOT / "requirements.txt")
    .read_text(encoding="utf-8")
    .splitlines()
)

# ---------------------------------------------------------------------- #
#  Package metadata
# ---------------------------------------------------------------------- #
setup(
    name="find_my_vibe",
    version="0.2.2",
    description="Multi-attribute fashion-image matcher (CLIP + Faiss)",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Huijing Yi, Jingyi Chen, Chenxu Lan, Wenyue Zhu",
    author_email=(
        "yi.hu@northeastern.edu, "
        "chen.jingyi6@northeastern.edu, "
        "lan.che@northeastern.edu, "
        "zhu.weny@northeastern.edu"
    ),
    license="MIT",
    python_requires=">=3.9, <3.13",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    packages=find_packages(exclude=("tests", "docs", "notebooks")),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
)
