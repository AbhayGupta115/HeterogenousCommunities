<h1 align="center">
Heterogeneous communities simulations <br/>  
</h1>

<p align="center">
<i>Both mean-field and numerical simulation (using EoN) for a planted partition network model with heterogeneous node susceptibility and transmissibility,</i>
</p>

<p align="center">

<a href="https://github.com/AbhayGupta115/HeterogenousCommunities/blob/main/LICENSE.md" target="_blank">
<img alt="License: MIT" src="https://img.shields.io/github/license/kaiser-dan/REPO">
</a>

<a href="https://www.python.org/" target="_blank">
<img alt="Made with Python" src="https://img.shields.io/badge/made%20with-python-1f425f.svg">
</a>

<a href="https://arxiv.org/abs/ARXIVID" target="_blank">
<img alt="ARXIV: ARXIVID" src="https://img.shields.io/badge/arXiv-ARXIVID-red.svg">
</a>

</p>

Research project exploring the effect of heterogeneous node susceptibility and transmissibility on the dynamics of spreading processes on community structured networks. The repository contains code for both mean-field analysis and numerical simulations using the [EoN](https://github.com/rivoal/EoN) library.

The repository contains the original scientific analyses developed by the Authors (see below) for the paper

- **(In preparation)** LastName, FirstName. Year. [Paper title](arxiv.org).

If you use this repository in any fashion in your own work, please cite our work according to the `CITATION.cff`.

# Contents

- [Contents](#contents)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
- [Usage](#usage)
  - [Reproducing experiments](#reproducing-experiments)
- [Other Information](#other-information)
  - [Built With](#built-with)
  - [Versioning](#versioning)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

# Getting Started

The code base for this project is written in Python.

These instructions will give you a copy of the project up and running on
your local machine for development, testing, and analysis purposes.

## Prerequisites

A compatible Python install is needed to begin - we used python 3.13.11 for this project.


## Installing

To (locally) reproduce this project, do the following:

1. Clone this repository. 
2. Create a virtual environment with the necessary packages. We provide a `environment.yml` file for this purpose.

This will install all necessary packages and place them into a virtual environment.


# Usage

## Reproducing experiments

Each figure in the paper has a corresponding Jupyter notebook. These notebooks can be used to reproduce the experiments and analyses for each figure. 

Each notebook also contains the code to produce the data for numerical simulations using EoN (if required). All these simulation were run on a high performance computing cluster, so the notebooks will not run in a reasonable time frame on a local machine. However, the notebooks can be adapted to run on a local machine if needed.


# Other Information

## Built With

### Repository and Version Control

- [ChooseALicense](https://choosealicense.com/) - Used to choose the license.
- [Commitizen](https://commitizen-tools.github.io/commitizen/) - Used to maintain Conventional Commit standard.

### Source code

- []()

### Research analysis

- [Snakemake]() - Used for organizing and running research pipeline.
- [Quarto]() - Used for writing analyses documents.


## Authors

All correspondence should be directed to [Abhay Gupta](abhay.gupta@virginia.edu).

- [Abhay Gupta](abhay.gupta@virginia.edu)
- [Nicholas W. Landry](nicholas.landry@virginia.edu)

## License

This project is licensed under the [MIT License](LICENSE.md) Creative Commons License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments
- We thank **Daniel Kaise** for copying the readme template from **Bllie Thompson** and providing it to us.