# SignSight

SignSight is a real-time webcam ASL alphabet recognizer using computer vision
and deep learning with PyTorch.

## Installation

First, ensure that Python 3.12 is installed.

```bash
# On Mac/Linux:
$ command -v python3.12
/usr/bin/python3.12

# On Windows:
$ py -3.12 --version
Python 3.12.13
```

Follow these steps in order by running the provided commands in your terminal.

### Setup

Download a clone of this repo on your machine and make it the working directory.

```bash
git clone https://github.com/van-riper/signsight.git
cd signsight
```

Create a virtual environment and switch to its Python interpreter.

```bash
# On Mac/Linux:
python3.12 -m venv .venv
source .venv/bin/activate

# On Windows:
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Make sure the current working directory is your cloned repo and the virtual
environment is activated before proceeding.

### Install Requirements

First, install PyTorch and TorchVision.

> **Please make sure** to specify the CPU package index if your machine does not
> have a GPU that is compatible with CUDA.

```bash
# If your machine has a CUDA-capable GPU:
pip install torch torchvision

# If not, use the CPU-only package index:
pip install --index-url=https://download.pytorch.org/whl/cpu torch torchvision
```

Then, install the remaining required packages.

```bash
pip install -r requirements.txt
```

#### Troubleshooting Cached Packages

Sometimes, `pip` will install packages from its cache if the packages have
previously been downloaded to your machine before. This can create problems if
the cache contains versions of packages that are not compatible with this
program. However, this can be avoided by forcing pip to download installed
packages instead of taking them out of the package cache on your machine.

If pip is not properly installing the required packages or the program won't
start, try by deleting this repo clone and redoing the installation process
from the beginning, except pass `--no-cache-dir` when running `pip install`.

```bash
# Example:
pip install --no-cache-dir -r requirements.txt
```

### Download the Dataset

This program's deep learning model is configured for the ASL Alphabet dataset
by Akash Nagaraj. To train the model on your machine, follow the steps below.

1. Go to the
   [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
   dataset on Kaggle.
2. Click the "Download" button and click the "Download dataset as zip" option.
3. Extract `archive.zip` in your downloads folder.
4. Move the extracted folder `archive/` into the folder `data/` in this repo.

> TODO: support kagglehub integration
