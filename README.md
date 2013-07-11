OpenCL Level Set Segmentation
==========================================
This is an implementation of level set image segmentation using OpenCL.
The speed function is defined as `-alpha*(epsilon-|T-intensity|)+(1-alpha)*curvature`.

See http://www.thebigblob.com/level-set-segmentation-on-gpus-using-opencl/ for more information on level sets and this implementation.

See LICENSE file for license information.

Dependencies
------------------------------
* OpenCL (You need an OpenCL implementation installed (AMD, NVIDIA, Intel, Apple...)
* Submodules SIPL and OpenCLUtilities (no need to install separately, however remember to initialize them "git submodule update" if you are cloning this repo)

Compiling
------------------------------
Use CMake and the provided CMakeLists.txt file to compile this program.
To compile on the run the example on Ubuntu, do the following:
```bash
mkdir build
cd build
cmake ../CMakeLists.txt
make
./levelSetSeg ../example_data/example.mhd result.mhd 10 50 40 5 1000 255 200 0.1 50 100
```

Usage
------------------------------

The arguments for the program are:
`./levelSetSeg inputFile.mhd outputFile.mhd seedX seedY seedZ seedRadius iterations threshold epsilon alpha [level window]`

inputFile.mhd is the metadata file of the volume you want to process and outputFile.mhd the path to where the result should be stored.
seedX, seedY, seedZ defines a spherical seed point with a radius seedRadius.
Iterations is the total number of iterations and threshold, epsilon, alpha are parameters for the speed function `-alpha*(epsilon-|T-intensity|)+(1-alpha)*curvature`.
If the level and window arguments are set, the segmentation result will be displayed as an overlay to the input volume.

Run with provided example data: `./levelSetSeg example_data/example.mhd result.mhd 10 50 40 5 1000 255 200 0.1 50 100`
