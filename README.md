OpenCL Level Set Segmentation
==========================================
This is a straightforward implementation of Level set image segmentation using OpenCL.
The speed function is defined as -alpha*(epsilon-(T-intensity))+(1-alpha)*curvature.

See LICENSE file for license information.

Dependencies
------------------------------
* OpenCL (You need an OpenCL implementation installed (AMD, NVIDIA, Intel, Apple...)
* Submodules SIPL and OpenCLUtilities (no need to install separately, however remember to initialize them "git submodule update" if you are cloning this repo)

Compiling
------------------------------
Use CMake to compile this program.
For instance, on Ubuntu do the following:
cmake .
make

Usage
------------------------------
./levelSeg inputFile.mhd outputFile.mhd seedX seedY seedZ seedRadius iterations threshold epsilon alpha [level window]