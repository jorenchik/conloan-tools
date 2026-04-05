# Getting started

## Python version

The package was tested on Python v3.11.14. For optimal compatibility and
performance, I recommend using Python v3.11.x. Using versions significantly
newer or older may lead to compatibility issues with native dependencies, 
specific PyTorch binaries or other packages.

### Managing versions with pyenv (Linux/macOS)

If your system Python is a different version, I recommend using pyenv to
manage your environment.

First make sure your environment has all the necessary prerequisites in order to install pyenv 
(see [suggested build environment](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)).
Then follow the official instruction of getting pyenv (see [getting pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#a-getting-pyenv)).

Make sure to add this to your environment. Example for bash.

```
# ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Once pyenv is installed, get the needed version.

```
# Install the recommended version
pyenv install 3.11.14

# Set it locally for this project
pyenv local 3.11.14
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

Before working with corpus oriented modules, you would need to setup Corpus
Workbench (CWB) utilities and add the corpus to the registry. See [corpus setup
instructions](corpus_setup.md).

