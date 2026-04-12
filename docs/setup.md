# Setting up 

## The library 

### Python version

The package was tested on Python v3.11.14. For optimal compatibility and
performance, I recommend using Python v3.11.x. Using versions significantly
newer or older may lead to compatibility issues with native dependencies, 
specific PyTorch binaries or other packages.

#### Managing versions with pyenv (Linux/macOS)

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

(Optional) It is recommended to create a virtual python environment.

```
python -m venv .venv
source .venv/bin/activate
```

### (Optional) Model related libraries

Firstly, to use ML model related commands, we would need to get 
pytorch if you need a specific version.

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Next, to use library fully, you'd need to install `models` extras.

```
pip install conloan-tools[models]
```

### Install

Use pip to install `conloan-tools`.

```
pip install conloan-tools
```


## Corpus

Before working with corpus oriented modules, you would need to setup Corpus
Workbench (CWB) utilities and add the corpus to the registry. Corpus Workbench
offers various tools for handling and querying `.vert` (corpus files). Conloan
uses `cwb` and the `cqp` utility specifically to query the sentences from a
corpus.

### Converting OPUS to VERT (CWB)

TODO.

### Install

To get the official release of `cwb`, refer to the official [install
page](https://cwb.sourceforge.io/install.php) (it contains simple instructions
to get the package on your system). The workbench is currently available for
following Linux distros:

* Debian, Ubuntu, Linux Mint, and derivatives;
* Fedora, Red Hat, and derivatives;
* Arch Linux, Manjaro, and friends.

_CWB does not have a native Windows build. To use it on Windows, you must use
Windows Subsystem for Linux (WSL). You can still use Conloan Tools excluding
the corpus and annotation (limited usage) packages._

### Setting up the registry

Once you obtain the `vert`. file of your corpus, you will need to encode it for
Corpus Workbench into the registry. The registry is a directory containing
plain-text definitions for each corpus. The data (corpus) encoding lives
separately. 

**Create the directory**

You may choose where you want to store your registry and data. CWB has defaults
set to `/usr/local/share/cwb/registry` and `/usr/local/share/cwb/data`. A good
option is to store it under `~/.local/share/cwb` for your user, which leads to
less complications due to file permissions.

_TODO: define CORPUS_DATA in the script._

```bash
mkdir -p /home/<user>/.local/share/cwb/registry
```

**Configure the environment**

CWB tools will look for the `CORPUS_REGISTRY` variable. Add this to your shell
setup file (e.g., `.bashrc` or `.zshrc`):

```bash
export CORPUS_REGISTRY="/home/<user>/.local/share/cwb/registry"
export CORPUS_DATA="/home/<user>/.local/share/cwb/data"
```

_Note that `CORPUS_DATA` is a non-standard environment variable which is used in
this package. By default Conloan Tools assumes the data directory right besides
the `CORPUS_REGISTRY` set directory._


### Adding a corpus


Encode `<my_corpus>.vert` as `<MYCORPUS>` (uppercase) / <mycorpus> (lowercase only):

```
mkdir -p $CORPUS_DATA/<MYCORPUS>
cwb-encode -d "$CORPUS_DATA/<MYCORPUS>" -f <my_corpus>.vert -R "$CORPUS_REGISTRY/<mycorpus>" \
  -c utf8 -x \
  -P pos -P lemma \
  -S doc:0+id+reference+section \
  -S p:0 -S s:0 -S g:0
cwb-makeall -V <MYCORPUS> 
```

_TODO: explain the command._

Finally, lets test our corpus out by quering the sentence that contains the token on
position of 120th index (121st token).

```
cqp
[no corpus]>MYCORPUS;
[MYCORPUS]>set Context s; 
[MYCORPUS]>[_.pos=120];
```

### Frequency list

Surprisal value detection usses Witten-Bell smoothing that builds itself from a
frequency list. To generate this file for your corpus, run the following:

```
cwb-lexdecode -f -s <MYCORPUS> > <my_corpus>_freq.txt
```
