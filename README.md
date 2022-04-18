# ODSC 2022: An introduction to drift detection

This repo contains the companion notebook for our [ODSC 2022 workshop](https://odsc.com/speakers/an-introduction-to-drift-detection/) on drift detection.

## Instructions

### Running on Colab

The easiest way to get started is to open up the workshop notebook in Google Colab by [clicking here](https://colab.research.google.com/github/ascillitoe/odsc_workshop/blob/main/intro_to_drift_detection_master.ipynb).

The notebook will run most efficiently on a GPU. To select a GPU runtime on Colab navigate to ***Runtime -> Change runtime type*** and select ***GPU*** under ***Hardware accelerator***.

### Running locally

The notebook can be run locally by first cloning this repo:

```
git clone https://github.com/ascillitoe/odsc_workshop.git
cd odsc_workshop
```

Then install all requirements:
```
pip install -r requirements.txt
```

Open the notebook:
```
jupyter-notebook intro_to_drift_detection.ipynb
```

## Requirements

You will require a working Python installation (version 3.7 - 3.9) and `jupyter` (installed with the above `pip install`).
