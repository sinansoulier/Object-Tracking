# Object-Tracking

## Introduction

This is the repository containing my source code and report for the final project of the course "ML Visual Object Tracking" at EPITA. The goal of this project is to implement a tracking algorithm that is able to track a person in a video sequence. The algorithm should be able to track the person in the video sequence, using IoU, Hungariam algorithm, Kalman filter and Deep Learning models.

## Installation

### Requirements

All the requirements are listed in the file `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Structure

* The code can be run in the following Jupiter notebook: [Main Jupyter notebook](./IoU_tracker.ipynb)

* The core of the trackers is in the [src](./src) folder, in the file `tracker.py`. The file contains the implementation of the IoU tracker, the Hungarian tracker, the Kalman tracker, etc.

* The Kalman filter is implemented in the file `KalmanFilter.py` in the folder [src](./src). To launch the material from the 1st practical session, you can run the file `objTracking.py` in the root of the repository.

```
python objTracking.py
```

* The [utils](./src/utils/) folder contains many utility functions used in the project. It also countains core utility functions for the trackers.

* The report is at the root of the repository: [Report](./report_MLVOT.pdf)