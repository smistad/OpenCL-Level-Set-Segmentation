#include "SIPL/Core.hpp"
#include "OpenCLUtilities/openCLUtilities.hpp"
#include <cmath>
#include <queue>
#include <iostream>
#include <cfloat>
using namespace SIPL;
using namespace std;

#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a < b ? a : b)

//#ifndef KERNELS_DIR
#define KERNELS_DIR "/home/smistad/Dropbox/Programmering/OpenCL-Level-Set-Segmentation/"
//#endif

typedef struct OpenCL {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Device device;
    cl::Platform platform;
} OpenCL;
Volume<float> * calculateSignedDistanceTransform(Volume<float> * phi) {
    // Identify the zero level set
    queue<int3> queue;
    Volume<float> * newPhi = new Volume<float>(phi->getSize());
    float inf = -9999999.0f;
    float inf2 = -99999999.0f;
    newPhi->fill(inf);
    float threshold = 0.00001f;

    for(int z = 1; z < phi->getDepth()-1; z++) {
    for(int y = 1; y < phi->getHeight()-1; y++) {
    for(int x = 1; x < phi->getWidth()-1; x++) {
        int3 pos(x,y,z);
        if(phi->get(pos) > 0) {
            // Check to see if it is a border point
            bool positive = false;
            bool negative = false;
            for(int a = -1; a < 2; a++) {
            for(int b = -1; b < 2; b++) {
            for(int c = -1; c < 2; c++) {
                int3 r(a,b,c);
                int3 n = pos + r;
                if(!phi->inBounds(n))
                    continue;
                if(phi->get(n) >= 0) {
                    positive = true;
                } else {
                    negative = true;
                }
            }}}

            if(positive && negative) {
                // Is a border point
                newPhi->set(pos, 0.0f);

                // Add neighbors to queue
                for(int a = -1; a < 2; a++) {
                for(int b = -1; b < 2; b++) {
                for(int c = -1; c < 2; c++) {
                    int3 r(a,b,c);
                    int3 n = pos + r;
                    if(!phi->inBounds(n))
                        continue;
                    queue.push(n);
                }}}
            }
        }
    }}}

    // Do a BFS over the entire volume with the zero level set as start points
    // If phi is negative, distance is set as negative
    // If it is positive distance is set as positive
    while(!queue.empty()) {
        int3 current = queue.front();
        queue.pop();
        if(newPhi->get(current) > inf) // already processed
            continue;

        bool negative = phi->get(current) < 0;
        float newDistance = inf;
        if(!negative)
            newDistance *= -1.0f;

        // Check all neighbors that are not inf or inf2
        for(int a = -1; a < 2; a++) {
        for(int b = -1; b < 2; b++) {
        for(int c = -1; c < 2; c++) {
            int3 r(a,b,c);
            int3 n = current + r;
            if(!phi->inBounds(n) || (a == 0 && b == 0 && c == 0))
                continue;

            if(newPhi->get(n) != inf && newPhi->get(n) != inf2) {
                if(negative) {
                    newDistance = MAX(newDistance, newPhi->get(n)-r.length());
                } else {
                    newDistance = MIN(newDistance, newPhi->get(n)+r.length());
                }
            } else if(newPhi->get(n) != inf2) {
                // Unvisited, Add to queue
                queue.push(n);
                newPhi->set(n,inf2);
            }
        }}}

        newPhi->set(current, newDistance);

    }

    delete phi;
    return newPhi;
}

void updateLevelSetFunction(OpenCL &ocl, cl::Kernel &kernel, cl::Image3D &input, cl::Image3D &phi_read, cl::Image3D &phi_write, int3 size) {

    kernel.setArg(0, input);
    kernel.setArg(1, phi_read);
    kernel.setArg(2, phi_write);

    ocl.queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
    );
}

void updateLevelSetFunction(OpenCL &ocl, cl::Kernel &kernel, cl::Image3D &input, cl::Image3D &phi_read, cl::Buffer &phi_write, int3 size) {

    kernel.setArg(0, input);
    kernel.setArg(1, phi_read);
    kernel.setArg(2, phi_write);

    ocl.queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
    );
}


void visualize(Volume<float> * input, Volume<float> * phi) {
    Volume<float2> * maskAndInput = new Volume<float2>(input->getSize());
    for(int i = 0; i < input->getTotalSize(); i++) {
        float2 v(0,0);
        float intensity = input->get(i);
        if(intensity < 50) {
            intensity = 0.0f;
        } else if(intensity > 150) {
            intensity = 1.0f;
        } else {
            intensity = (intensity - 50.0f)/100.0f;
        }
        v.x = intensity;
        if(phi->get(i) < 0) {
            v.y = 1.0f;
        }
        maskAndInput->set(i,v);
    }
    maskAndInput->show();

}

Volume<float> * runLevelSet(OpenCL &ocl, Volume<float> * input, int3 seedPos, float seedRadius, int iterations, int reinitialize) {
    int3 size = input->getSize();
    cl::Image3D inputData = cl::Image3D(
            ocl.context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            cl::ImageFormat(CL_R, CL_FLOAT),
            input->getWidth(),
            input->getHeight(),
            input->getDepth(),
            0,0,
            input->getData()
    );

    cl::Image3D phi_1 = cl::Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_R, CL_FLOAT),
            input->getWidth(),
            input->getHeight(),
            input->getDepth()
    );

    // Create seed
    cl::Kernel createSeedKernel(ocl.program, "initializeLevelSetFunction");
    createSeedKernel.setArg(0, phi_1);
    createSeedKernel.setArg(1, seedPos.x);
    createSeedKernel.setArg(2, seedPos.y);
    createSeedKernel.setArg(3, seedPos.z);
    createSeedKernel.setArg(4, seedRadius);
    ocl.queue.enqueueNDRangeKernel(
            createSeedKernel,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
    );

    cl::Kernel kernel(ocl.program, "updateLevelSetFunction");
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;

    if(ocl.device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") == 0) {
        // Create auxillary buffer
        cl::Buffer writeBuffer = cl::Buffer(
                ocl.context,
                CL_MEM_WRITE_ONLY,
                sizeof(float)*size.x*size.y*size.z
        );

        for(int i = 0; i < iterations; i++) {
            updateLevelSetFunction(ocl, kernel, inputData, phi_1, writeBuffer, size);
            ocl.queue.enqueueCopyBufferToImage(
                    writeBuffer,
                    phi_1,
                    0,
                    origin,
                    region
            );
        }
    } else {
        cl::Image3D phi_2 = cl::Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_R, CL_FLOAT),
            input->getWidth(),
            input->getHeight(),
            input->getDepth()
        );

        for(int i = 0; i < iterations; i++) {
            if(i % 2 == 0) {
                updateLevelSetFunction(ocl, kernel, inputData, phi_1, phi_2, size);
            } else {
                updateLevelSetFunction(ocl, kernel, inputData, phi_2, phi_1, size);
            }
        }
        if(iterations % 2 != 0) {
            // Phi_2 was written to in the last iteration, copy this to the result
            ocl.queue.enqueueCopyImage(phi_2,phi_1,origin,origin,region);
        }
    }

    Volume<float> * phi = new Volume<float>(input->getSize());
    float * data = phi->getData();
    ocl.queue.enqueueReadImage(
            phi_1,
            CL_TRUE,
            origin,
            region,
            0, 0,
            data
    );

    phi->setData(data);

    return phi;
}

int main(int argc, char ** argv) {

    if(argc < 8) {
        cout << "usage: " << argv[0] << " inputFile.mhd outputFile.mhd seedX seedY seedZ seedRadius iterations" << endl;
        return -1;
    }

    // Create OpenCL context
    OpenCL ocl;
    ocl.context = createCLContextFromArguments(argc,argv);
    VECTOR_CLASS<cl::Device> devices = ocl.context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    ocl.device = devices[0];
    ocl.queue = cl::CommandQueue(ocl.context, devices[0]);
    string filename = string(KERNELS_DIR) + string("kernels.cl");
    string buildOptions = "";
    if(ocl.device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") == 0)
        buildOptions = "-DNO_3D_WRITE";
    ocl.program = buildProgramFromSource(ocl.context, filename);

    // Load volume
    Volume<float> * input = new Volume<float>(argv[1]);
    float3 spacing = input->getSpacing();

    std::cout << "Dataset of size " << input->getWidth() << ", " << input->getHeight() << ", " << input->getDepth() << " loaded "<< std::endl;

    // Set initial mask
    int3 origin(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    float seedRadius = atof(argv[6]);

    // Do level set
    try {
        Volume<float> * res = runLevelSet(ocl, input, origin, seedRadius, atoi(argv[7]), 1000000);

        // Visualize result
        visualize(input, res);

        // Store result
        Volume<char> * segmentation = new Volume<char>(res->getSize());
        segmentation->setSpacing(spacing);
        for(int i = 0; i < res->getTotalSize(); i++) {
            if(res->get(i) < 0.0f) {
                segmentation->set(i, 1);
            } else {
                segmentation->set(i, 0);
            }
        }

        segmentation->save(argv[2]);

    } catch(cl::Error &e) {
        cout << "OpenCL error occured: " << e.what() << " " << getCLErrorString(e.err()) << endl;
    }

}
