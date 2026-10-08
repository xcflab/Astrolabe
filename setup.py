from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup


setup(
    name="Astrolabe",
    version="0.1.0",
    description="Programmable all-atom biomolecular design through cross-context coordination and structural intervention",
    author="Zhengpeng Zeng, Chunfu Xu",
    url="https://github.com/Molaison/Astrolabe",
    project_urls={
        "Documentation": "https://github.com/Molaison/Astrolabe/blob/main/docs/index.md",
        "Source": "https://github.com/Molaison/Astrolabe",
        "Issues": "https://github.com/Molaison/Astrolabe/issues",
    },
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.11",
    # Ship the library and paper adapters alongside their portable configuration
    # and input resources (resolved relative to astrolabe.config_dir()).
    packages=find_packages(include=["astrolabe", "astrolabe.*"])
    + find_namespace_packages(include=["config", "config.*", "resources",
                                       "data", "data.framework_fastas", "data.framework_homologies",
                                       "examples.paper", "examples.paper.*"]),
    py_modules=["run"],
    include_package_data=True,
    install_requires=[
        "absl-py>=2.1,<3",
        "biopython>=1.84,<2",
        "filelock>=3.12,<4",
        "hydra-core>=1.3,<1.4",
        "matplotlib>=3.8,<4",
        "numpy>=1.26,<3",
        "omegaconf>=2.3,<2.4",
        "prompt-toolkit>=3.0,<4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "build>=1",
            "setuptools>=70.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "astrolabe=astrolabe.cli:main",
        ]
    },
)
