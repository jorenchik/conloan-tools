
# conloan-tools

A software pipeline for borrowing (loanwords), code-switching data extraction,
and classifier training. `conloan-tools` provides a collection of CLI tools to
perform tasks like extracting CWB compliant corpora, generating annotation
sheets, training and evaluating sequence classification models, and managing
datasets.

This repository is an artifact of the bachelor thesis pipeline for the
`ConLoan-LV` dataset.

## Features

The core pipeline consists of three primary functional groups:

-   **Annotation** (`conloan-tools annotation`): JSON format utilities, dataset
    translation via instruction-based or specialized models, annotation sheet
    generation from lemmas, and validation steps.
-   **Classifier** (`conloan-tools classifier`): Tools to train a classifier on
    the dataset, k-fold evaluate splits, inspect the results or tokenization.
-   **Corpus** (`conloan-tools corpus`): Utilities to build, convert, and
    inspect validating HDF5 corpus indices and retrieving sentence examples
    (containing code-switching sequences, lemmas, and named entities) for
    annotation. Requires Corpus Workbench (CWB) for the query functionality.

## Installation

The package requires **Python 3.11.x**. Compatibility with older/newer versions
is not guaranteed due to specific PyTorch, Transformers and their dependency
specifics.

For detailed setup steps, please refer to the [setup guide](docs/setup.md).

### 1. Python Package

Install the base package (omits heavy neural network dependencies, but does not
have the complete functionality):

```bash
pip install conloan-tools
```

To utilize the ML models and classifiers, install the model extras and a
compatible PyTorch version. E.g., for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install conloan-tools[models]
```

### 2. Corpus Workbench (CWB) Dependency

For detailed setup steps, please refer to the [setup guide](docs/setup.md).

## CLI Usage

The entry point for the CLI is `conloan-tools`. Use `--help` on any subgroup to
see specific arguments.

```bash
TODO: example.
```

## Citation

If you use `conloan-tools` or the associated `ConLoan-LV` dataset in your
research, please cite the respective:

```bibtex
# Dataset itself.
@misc{stekels-2026-clarin-dataset,
    title = {{ConLoan}-{LV}: A Contrastive Dataset for Latvian Language Loanwords, Code-switching, and Named Entities},
    author = {{\v S}tekeļs, Jorens},
    url = {http://hdl.handle.net/20.500.12574/158},
    note = {{CLARIN}-{LV} digital library at {IMCS}, University of Latvia},
    copyright = {Creative Commons - Attribution 4.0 International ({CC} {BY} 4.0)},
    year = {2026} 
}

# (and/or) This pipeline.
@software{stekels-2026-tools,
  author       = {Jorens Štekeļs},
  title        = {conloan-tools: {A} software pipeline for borrowing and code-switching data extraction},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  url          = {https://github.com/jorenchik/conloan-tools},
  version      = {1.0.0}
}

# (and/or) The thesis.
@thesis{stekels2026contextual,
  author       = {Štekeļs, Jorens},
  title        = {Kontekstuāla pieeja latviešu valodas aizguvumu noteikšanā: datu kopas veidošana un klasifikācijas eksperimenti},
  school       = {Latvijas Universitāte, Eksakto zinātņu un tehnoloģiju fakultāte},
  year         = {2026},
  address      = {Rīga, Latvija},
  type         = {Bakalaura darbs},
  author_id    = {js21283},
  supervisor   = {Dr. dat. Normunds Grūzītis},
  note         = {English title: Contextual Approach to Latvian Loanword Detection: Dataset Creation and Classification Experiments},
  url          = {http://hdl.handle.net/20.500.12574/158}
}
```
