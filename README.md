# SignSight

SignSight is a real-time webcam ASL alphabet recognizer using computer vision
and deep learning with PyTorch.

## Setup

Requires Python version 3.12:

```bash
$ python --version
Python 3.12.13
```

### Clone Repo

```bash
git clone https://github.com/van-riper/signsight.git
cd signsight
```

### Create and Activate Virtual Environment

On Mac/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### Install Requirements

```bash
pip install --no-cache-dir -r requirements.txt
pip install --index-url=https://download.pytorch.org/whl/cpu torch torchvision
```
