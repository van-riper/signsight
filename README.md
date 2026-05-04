# SignSight

SignSight is a real-time webcam ASL alphabet recognizer using computer vision
and deep learning with PyTorch.

Python 3.12 is required to run this program, so make sure it is installed and
available on your system.

This program uses the following packages:

- [PyTorch](https://pypi.org/project/torch/)
- [TorchVision](https://pypi.org/project/torchvision/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- [NumPy](https://pypi.org/project/numpy/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)

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

### Download Datasets and Landmarks

This program's deep learning model is configured for the ASL-HG dataset, which
you can read more about in this
[NIH Data Brief](https://pmc.ncbi.nlm.nih.gov/articles/PMC12877850/).

Additionally, the inference pipeline relies on the MediaPipe hand landmarker
task. More info can be found in this
[Google AI Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

#### Dataset

To download the dataset onto your machine, follow these steps:

1. Download the dataset at the
   [download page](https://data.mendeley.com/datasets/j4y5w2c8w9/1).
2. Extract the zip file in your downloads folder.
3. Move the subfolder `ASL_HG_36000/`into the folder `data/` in this repo.
4. Extract both `ASL_Processed_Images.zip` and `ASL_Raw_Images.zip` inside the
   `ASL_HG_36000/` folder. There should now be two folders called `asl_dataset/`
   and `asl_processed/` within `ASL_HG_36000/`.

Your `data/` folder should look like this:

```text
data/
├── ASL_HG_3600/
│   ├── asl_dataset/
│   │   └── ...
│   ├── asl_processed/
│   │   └── ...
│   ├── ASL_Processed_Images.zip
│   └── ASL_Raw_images.zip
└── .gitkeep
```

#### Landmarks

To download the MediaPipe task model, open this link:
[hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

Then, move `hand_landmarker.task` into the `model/` folder in this repo.

## Usage

### Model Training and Evaluation

After installation, the deep learning network can begin training and evaluation.

First, start by making sure the correct Python interpreter is selected.

```bash
$ python --version
Python 3.12.13
```

Then, start the training pipeline.

```bash
python -m signsight train
```

This will take a while to run, especially if your machine doesn't have a GPU.
The progress of the current epoch's batch cycling is displayed in real-time so
you can monitor how fast the model takes to train.

The training and evaluation batch size is preset to either **32** or **64**. If
SignSight detects a CUDA device on your machine, the batch size is set to 64.
If not, it defaults to 32.

If you want to specify the batch size, pass the `--batch-size` flag to override
the preset defaults.

```bash
# Example: set batch size to 128
python -m signsight --batch-size=128 train
```

Once the model has been trained, run the model evaluation. Make sure to pass the
`--batch-size` flag if you want to evaluate using a different batch size.

```bash
python -m signsight eval
```

Then confirm the model's accuracy is high enough (should be at least 98%).

This will also bring up a confusion matrix plot to show more details on the
model's accuracy for each dataset class.

If your machine cannot display the plot, you can still view it by opening the
image `confusion_matrix.png`, which get written to disk after the model
evaluation is complete.

### Inference Interface

After the model has been trained, you can run the inference interface.

```bash
python -m signsight run
```

This will turn on your webcam and feed it into the inference pipeline. Your
webcam feed will open up in a new window.

Try holding your hand to the camera, you should see the landmarkers rendered
onto your hand. Then try signing ASL letters, you should see the model's
prediction drawn on the top right corner of the window.

To quit the inference interface, press "Q" and the program will exit.

## Troubleshooting

### Pip Cached Packages

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
