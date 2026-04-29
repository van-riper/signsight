# SignSight

SignSight is a real-time webcam ASL alphabet recognizer using computer vision
and deep learning with PyTorch.

## Installation

Requires Python version 3.12 exactly.

```bash
$ python --version
Python 3.12.13
```

### Setup

Download a clone of the repo on your machine and make it the working directory.

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

First, install PyTorch and TorchVision. **Please make sure** to specify the CPU
package index if your machine does not have a GPU that is compatible with CUDA.

```bash
# If your machine has a CUDA-capable GPU:
pip install torch torchvision

# If not, use the CPU-only package index:
pip install --index-url=https://download.pytorch.org/whl/cpu torch torchvision
```

Then, install the rest of the required packages:

```bash
pip install -r requirements.txt
```
