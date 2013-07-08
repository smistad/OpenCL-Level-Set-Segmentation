#include "SIPL/Core.hpp"
using namespace SIPL;

Volume<float> * calculateSignedDistanceTransform(Volume<float> * phi) {

}

Volume<float> * createInitialMask(int3 origin, int size) {

}

Volume<float> * updateLevelSetFunction(Volume<float> * phi) {
    Volume<float> * phiNext = new Volume<float>(phi->getSize());
    for(int z = 1; z < phi->getDepth()-1; < z++) {
    for(int y = 1; y < phi->getHeight()-1; < y++) {
    for(int x = 1; x < phi->getWidth()-1; < x++) {
        int3 pos(x,y,z);

        // Calculate all first order derivatives
        float3 D;
        float3 Dminus;
        float3 Dpluss;

        // Calculate all second order derivatives

        // Calculate normals

        // Calculate gradient
        float3 gradient;

        // Calculate curvature

        // Calculate speed term

        float alpha;
        float threshold;
        float epsilon;
        float speed;

        // Update the level set function phi
        phiNext->set(pos, phi->get(pos) + speed*gradient.length());
    }}}
    delete phi;
    return phiNext;
}

Volume<char> * runLevelSet(Volume<short> * input, Volume<float> * initialMask, int iterations, int reinitialize) {
    Volume<float> * phi = calculateSignedDistanceTransform(initialMask);

    for(int i = 0; i < iterations; i++) {
        phi = updateLevelSetFunction(phi);

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
