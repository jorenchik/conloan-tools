# Setting up the corpus(/-ra)

Corpus Workbench offers various tools for handling and querying `.vert` (corpus
files). Conloan uses `cwb` and the `cqp` utility specifically to query the
sentences from a corpus.

## Converting OPUS to VERT (CWB)

TODO.

## Install

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

## Setting up the registry

Once you obtain the `vert`. file of your corpus, you will need to encode it for
Corpus Workbench into the registry. The registry is a directory containing
plain-text definitions for each corpus. The data (corpus) encoding lives
separately. 

### Create the directory

You may choose where you want to store your registry and data. CWB has defaults
set to `/usr/local/share/cwb/registry` and `/usr/local/share/cwb/data`. A good
option is to store it under `~/.local/share/cwb` for your user, which leads to
less complications due to file permissions.

_TODO: define CORPUS_DATA in the script._

```bash
mkdir -p /home/<user>/.local/share/cwb/registry
```

### Configure the environment

CWB tools will look for the `CORPUS_REGISTRY` variable. Add this to your shell
setup file (e.g., `.bashrc` or `.zshrc`):

```bash
export CORPUS_REGISTRY="/home/<user>/.local/share/cwb/registry"
export CORPUS_DATA="/home/<user>/.local/share/cwb/data"
```

_Note that `CORPUS_DATA` is a non-standard environment variable which is used in
this package. By default Conloan Tools assumes the data directory right besides
the `CORPUS_REGISTRY` set directory._


## Adding a corpus


Encode `<my_corpus>.vert` as `<MYCORPUS>`:

```
mkdir $CORPUS_DATA/<MYCORPUS>
cwb-encode -d "$CORPUS_DATA/<MYCORPUS>" -f <my_corpus>.vert -R "$CORPUS_REGISTRY" \
  -c utf8 -x \
  -P pos -P lemma \
  -S doc:0+id+reference+section \
  -S p:0 -S s:0 -S g:0
```

_TODO: explain the command._

Lets test our corpus out by quering the sentence that contains the token on
position of 120th index (121st token).

```
cqp
[no corpus]>MYCORPUS;
[MYCORPUS]>set Context s; 
[MYCORPUS]>[_.pos=120];
```

## Frequency list

Surprisal value detection usses Witten-Bell smoothing that builds itself from a
frequency list. To generate this file for your corpus, run the following:

```
cwb-lexdecode -f -s <MYCORPUS> > <my_corpus>_freq.txt
```
