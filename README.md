# IncrementalVisualTracker

This repository contains C++ implementation of Incremental Visual Tracking algorithm presented in the paper [Incremental Learning for Robust Visual Tracking](http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf).
The code was mostly grounded on the original MATLAB-implementation posted in http://www.cs.toronto.edu/~dross/ivt/.

## Requirements
- **OpenCV** >= 3.0 with Eigen support
- **Eigen3**

## Run
Inside build folder execute
```
./ivt --input <path_to_video>
```