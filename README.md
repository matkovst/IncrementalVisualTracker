# IncrementalVisualTracker

This repository contains C++ implementation of Incremental Visual Tracking algorithm presented in the paper [Incremental Learning for Robust Visual Tracking](http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf).
The code was mostly grounded on the original MATLAB-implementation posted in http://www.cs.toronto.edu/~dross/ivt/.

## Requirements
- **OpenCV** >= 3.0 with Eigen support
- **Eigen3**

## Run
Cmake yields two executables *ivt_fast* and *ivt_accurate* which for operating F32 and F64 precision respectively. Choose appropriate mode and launch it inside build folder
```
./ivt_accurate --input <path_to_video>
```