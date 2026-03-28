# Getting started

## Python version

The package was tested on Python v3.11.14. For optimal compatibility and
performance, I recommend using Python v3.11.x. Using versions significantly
newer or older may lead to compatibility issues with native dependencies, 
specific PyTorch binaries or other packages.

### Managing versions with pyenv (Linux/macOS)

If your system Python is a different version, I recommend using pyenv to
manage your environment:

```
# Install the recommended version
pyenv install 3.11.14

# Set it locally for this project
pyenv local 3.11.14
```

To make the version change seemless, add `pyenv` initialization in your shell
startup script:

```
# ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

## (Optional) Model related libraries

Firstly, to use ML model related commands, we would need to get 
pytorch if you need a specific version.

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Next, to use library fully, you'd need to install `models` extras.

```
pip install conloan-tools[models]
```

## Install

Use pip to install `conloan-tools`.

```
pip install conloan-tools
```

## Preparing your corpus 

Before working with this module, we would need to setup Corpus Workbench
utilities and add the corpus to the registry. See [corpus setup
instructions](corpus_setup.md).
