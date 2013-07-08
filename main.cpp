#include "SIPL/Core.hpp"
#include <cmath>
using namespace SIPL;

#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a < b ? a : b)

Volume<float> * calculateSignedDistanceTransform(Volume<float> * phi) {

}

Volume<float> * createInitialMask(int3 origin, int size) {

}

Volume<float> * updateLevelSetFunction(Volume<short> * input, Volume<float> * phi) {
    Volume<float> * phiNext = new Volume<float>(phi->getSize());
    for(int z = 1; z < phi->getDepth()-1; z++) {
    for(int y = 1; y < phi->getHeight()-1; y++) {
    for(int x = 1; x < phi->getWidth()-1; x++) {
        int3 pos(x,y,z);

        // Calculate all first order derivatives
        float3 D(
                0.5f*(phi->get(int3(x+1,y,z))+phi->get(int3(x-1,y,z))),
                0.5f*(phi->get(int3(x,y+1,z))+phi->get(int3(x,y-1,z))),
                0.5f*(phi->get(int3(x,y,z+1))+phi->get(int3(x,y,z-1)))
        );
        float3 Dminus(
                phi->get(int3(x-1,y,z))-phi->get(pos),
                phi->get(int3(x,y-1,z))-phi->get(pos),
                phi->get(int3(x,y,z-1))-phi->get(pos)
        );
        float3 Dplus(
                phi->get(int3(x+1,y,z))-phi->get(pos),
                phi->get(int3(x,y+1,z))-phi->get(pos),
                phi->get(int3(x,y,z+1))-phi->get(pos)
        );

        // Calculate gradient
        float3 gradientMin(
                sqrt(pow(MIN(Dplus.x, 0.0f), 2.0f) + pow(MIN(-Dminus.x, 0.0f), 2.0f)),
                sqrt(pow(MIN(Dplus.y, 0.0f), 2.0f) + pow(MIN(-Dminus.y, 0.0f), 2.0f)),
                sqrt(pow(MIN(Dplus.z, 0.0f), 2.0f) + pow(MIN(-Dminus.z, 0.0f), 2.0f))
        );
        float3 gradientMax(
                sqrt(pow(MAX(Dplus.x, 0.0f), 2.0f) + pow(MAX(-Dminus.x, 0.0f), 2.0f)),
                sqrt(pow(MAX(Dplus.y, 0.0f), 2.0f) + pow(MAX(-Dminus.y, 0.0f), 2.0f)),
                sqrt(pow(MAX(Dplus.z, 0.0f), 2.0f) + pow(MAX(-Dminus.z, 0.0f), 2.0f))
        );

        // Calculate all second order derivatives
        float3 DxMinus(
                0.0f,
                0.5f*(phi->get(int3(x+1,y-1,z))+phi->get(int3(x-1,y-1,z))),
                0.5f*(phi->get(int3(x+1,y,z-1))+phi->get(int3(x-1,y,z-1)))
        );
        float3 DxPlus(
                0.0f,
                0.5f*(phi->get(int3(x+1,y+1,z))+phi->get(int3(x-1,y+1,z))),
                0.5f*(phi->get(int3(x+1,y,z+1))+phi->get(int3(x-1,y,z+1)))
        );
        float3 DyMinus(
                0.5f*(phi->get(int3(x-1,y+1,z))+phi->get(int3(x-1,y-1,z))),
                0.0f,
                0.5f*(phi->get(int3(x,y+1,z-1))+phi->get(int3(x,y-1,z-1)))
        );
        float3 DyPlus(
                0.5f*(phi->get(int3(x+1,y+1,z))+phi->get(int3(x+1,y-1,z))),
                0.0f,
                0.5f*(phi->get(int3(x,y+1,z+1))+phi->get(int3(x,y-1,z+1)))
        );
        float3 DzMinus(
                0.5f*(phi->get(int3(x-1,y,z+1))+phi->get(int3(x-1,y,z-1))),
                0.5f*(phi->get(int3(x,y-1,z+1))+phi->get(int3(x,y-1,z-1))),
                0.0f
        );
        float3 DzPlus(
                0.5f*(phi->get(int3(x+1,y,z+1))+phi->get(int3(x+1,y,z-1))),
                0.5f*(phi->get(int3(x,y+1,z+1))+phi->get(int3(x,y+1,z-1))),
                0.0f
        );

        // Calculate curvature
        float3 nMinus;
        float3 nPlus;
        float curvature = (nPlus.x-nMinus.x)+(nPlus.y-nMinus.y)+(nPlus.z-nPlus.z);

        // Calculate speed term

        float alpha;
        float threshold;
        float epsilon;
        float speed = alpha*(epsilon-fabs(input->get(pos)-threshold)) + (1.0f-alpha)*curvature;
        float3 gradient;
        if(speed < 0) {
            gradient = gradientMin;
        } else {
            gradient = gradientMin;
        }

        // Stability CFL
        // max(fabs(speed*gradient.length()))
        float deltaT = 1.0f / 1.0f ;

        // Update the level set function phi
        phiNext->set(pos, phi->get(pos) + deltaT*speed*gradient.length());
    }}}
    delete phi;
    return phiNext;
}

Volume<char> * runLevelSet(Volume<short> * input, Volume<float> * initialMask, int iterations, int reinitialize) {
    Volume<float> * phi = calculateSignedDistanceTransform(initialMask);

    for(int i = 0; i < iterations; i++) {
        phi = updateLevelSetFunction(input, phi);

        if(i > 0 && i % reinitialize == 0)
            phi = calculateSignedDistanceTransform(phi);
    }

}

int main(int argc, char ** argv) {

    // Load volume
    Volume<short> * input = new Volume<short>(argv[1]);

    // Set initial mask
    int3 origin(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    int size = atoi(argv[5]);

    Volume<float> * initialMask = createInitialMask(origin, size);

    // Do level set
    Volume<char> * segmentation = runLevelSet(input, initialMask, 100, 20);

    // Visualize result
}
