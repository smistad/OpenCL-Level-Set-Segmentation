#include "SIPL/Core.hpp"
#include "OpenCLUtilities/openCLUtilities.hpp"
#include <iostream>
using namespace SIPL;
using namespace std;

#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a < b ? a : b)

#ifndef KERNELS_DIR
#define KERNELS_DIR ""
#endif

typedef struct OpenCL {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Device device;
} OpenCL;

void updateLevelSetFunction(
        OpenCL &ocl,
        cl::Kernel &kernel,
        cl::Image3D &input,
        cl::Image3D &phi_read,
        cl::Image3D &phi_write,
        int3 size,
        float threshold,
        float epsilon,
        float alpha
        ) {

    kernel.setArg(0, input);
    kernel.setArg(1, phi_read);
    kernel.setArg(2, phi_write);
    kernel.setArg(3, threshold);
    kernel.setArg(4, epsilon);
    kernel.setArg(5, alpha);

    ocl.queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
    );
}

void updateLevelSetFunction(
        OpenCL &ocl,
        cl::Kernel &kernel,
        cl::Image3D &input,
        cl::Image3D &phi_read,
        cl::Buffer &phi_write,
        int3 size,
        float threshold,
        float epsilon,
        float alpha
        ) {

    kernel.setArg(0, input);
    kernel.setArg(1, phi_read);
    kernel.setArg(2, phi_write);
    kernel.setArg(3, threshold);
    kernel.setArg(4, epsilon);
    kernel.setArg(5, alpha);


    ocl.queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
    );
}


void visualize(Volume<float> * input, Volume<float> * phi, float level, float window) {
    Volume<float2> * maskAndInput = new Volume<float2>(input->getSize());
    for(int i = 0; i < input->getTotalSize(); i++) {
        float2 v(0,0);
        float intensity = input->get(i);
        if(intensity < level-window*0.5f) {
            intensity = 0.0f;
        } else if(intensity > level+window*0.5f) {
            intensity = 1.0f;
        } else {
            intensity = (float)(intensity-(level-window*0.5f)) / window;
        }

        v.x = intensity;
        if(phi->get(i) < 0) {
            v.y = 1.0f;
        }
        maskAndInput->set(i,v);
    }
    maskAndInput->show();

}

Volume<float> * runLevelSet(
        OpenCL &ocl,
        Volume<float> * input,
        int3 seedPos,
        float seedRadius,
        int iterations,
        float threshold,
        float epsilon,
        float alpha
        ) {
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
            updateLevelSetFunction(ocl, kernel, inputData, phi_1, writeBuffer, size, threshold, epsilon, alpha);
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
                updateLevelSetFunction(ocl, kernel, inputData, phi_1, phi_2, size, threshold, epsilon, alpha);
            } else {
                updateLevelSetFunction(ocl, kernel, inputData, phi_2, phi_1, size, threshold, epsilon, alpha);
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

    if(argc < 11) {
        cout << endl;
        cout << "OpenCL Level Set Segmentation by Erik Smistad 2013" << endl;
        cout << "www.github.com/smistad/OpenCL-Level-Set-Segmentation/" << endl;
        cout << "======================================================" << endl;
        cout << "The speed function is defined as -alpha*(epsilon-(T-intensity))+(1-alpha)*curvature" << endl;
        cout << "Usage: " << argv[0] << " inputFile.mhd outputFile.mhd seedX seedY seedZ seedRadius iterations threshold epsilon alpha [level window]" << endl;
        cout << "If the level and window arguments are set, the segmentation result will be displayed as an overlay to the input volume " << endl;
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
    int3 seedPosition(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    float seedRadius = atof(argv[6]);

    // Do level set
    try {
        Volume<float> * res = runLevelSet(ocl, input, seedPosition, seedRadius, atoi(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]));

        // Visualize result
        if(argc == 13) {
            float level = atof(argv[11]);
            float window = atof(argv[12]);
            visualize(input, res, level, window);
        }

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
        cout << "OpenCL error occurred: " << e.what() << " " << getCLErrorString(e.err()) << endl;
    }

}
