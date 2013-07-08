#include "SIPL/Core.hpp"
#include <cmath>
#include <queue>
#include <iostream>
using namespace SIPL;
using namespace std;

#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a < b ? a : b)

Volume<float> * calculateSignedDistanceTransform(Volume<float> * phi) {
    // Identify the zero level set
    queue<int3> queue;
    Volume<float> * newPhi = new Volume<float>(phi->getSize());
    float inf = -9999999.0f;
    float inf2 = -99999999.0f;
    newPhi->fill(inf);
    float threshold = 0.0001f;

    for(int z = 1; z < phi->getDepth()-1; z++) {
    for(int y = 1; y < phi->getHeight()-1; y++) {
    for(int x = 1; x < phi->getWidth()-1; x++) {
        int3 pos(x,y,z);
        if(fabs(phi->get(pos)) < threshold) {
            newPhi->set(pos, 0.0f);

            // Add neighbors to queue
            for(int a = -1; a < 2; a++) {
            for(int b = -1; b < 2; b++) {
            for(int c = -1; c < 2; c++) {
                int3 r(a,b,c);
                int3 n = pos + r;
                if(!phi->inBounds(n))
                    continue;
                if(fabs(phi->get(n)) >= threshold) {
                    queue.push(n);
                }
            }}}
        }
    }}}

    // Do a BFS over the entire volume with the zero level set as start points
    // If phi is negative, distance is set as negative
    // If it is positive distance is set as positive
    while(!queue.empty()) {
        int3 current = queue.front();
        queue.pop();

        bool negative = phi->get(current) < 0;
        float newDistance = inf;
        if(!negative)
            newDistance *= -1.0f;

        // Check all neighbors that are not inf
        for(int a = -1; a < 2; a++) {
        for(int b = -1; b < 2; b++) {
        for(int c = -1; c < 2; c++) {
            int3 r(a,b,c);
            int3 n = current + r;
            if(!phi->inBounds(n) || (a == 0 && b == 0 && c == 0))
                continue;

            if(newPhi->get(n) != inf && newPhi->get(n) != inf2) {
                if(negative) {
                    newDistance = MAX(newDistance, newPhi->get(n))-1.0f;
                } else {
                    newDistance = MIN(newDistance, newPhi->get(n))+1.0f;
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

Volume<float> * createInitialMask(int3 origin, int size, int3 volumeSize) {

    Volume<float> * mask = new Volume<float>(volumeSize);
    mask->fill(1.0f);

    for(int z = origin.z; z < origin.z+size; z++) {
    for(int y = origin.y; y < origin.y+size; y++) {
    for(int x = origin.x; x < origin.x+size; x++) {
        float value = -1.0f;
        if(z == origin.z || y == origin.y || x == origin.x
                || z == origin.z+size-1 || y == origin.y+size-1 || x == origin.x+size-1)
            value = 0.0f;
        mask->set(x,y,z, value);
    }}}

    return mask;
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
        float3 nMinus(
                Dminus.x / sqrt(Dminus.x*Dminus.x+pow(0.5f*(DyMinus.x+D.y),2.0f)+pow(0.5f*(DzMinus.x+D.z),2.0f)),
                Dminus.x / sqrt(Dminus.y*Dminus.y+pow(0.5f*(DxMinus.y+D.x),2.0f)+pow(0.5f*(DzMinus.y+D.z),2.0f)),
                Dminus.x / sqrt(Dminus.z*Dminus.z+pow(0.5f*(DxMinus.z+D.x),2.0f)+pow(0.5f*(DyMinus.z+D.y),2.0f))
        );
        float3 nPlus(
                Dplus.x / sqrt(Dplus.x*Dplus.x+pow(0.5f*(DyPlus.x+D.y),2.0f)+pow(0.5f*(DzPlus.x+D.z),2.0f)),
                Dplus.x / sqrt(Dplus.y*Dplus.y+pow(0.5f*(DxPlus.y+D.x),2.0f)+pow(0.5f*(DzPlus.y+D.z),2.0f)),
                Dplus.x / sqrt(Dplus.z*Dplus.z+pow(0.5f*(DxPlus.z+D.x),2.0f)+pow(0.5f*(DyPlus.z+D.y),2.0f))
        );

        float curvature = (nPlus.x-nMinus.x)+(nPlus.y-nMinus.y)+(nPlus.z-nPlus.z);

        // Calculate speed term
        float alpha = 0.5f;
        float threshold;
        float epsilon;
        float speed = alpha*(epsilon-fabs(input->get(pos)-threshold)) + (1.0f-alpha)*curvature;

        // Determine gradient based on speed direction
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

void runLevelSet(Volume<short> * input, Volume<float> * initialMask, int iterations, int reinitialize) {
    Volume<float> * phi = calculateSignedDistanceTransform(initialMask);

    for(int i = 0; i < iterations; i++) {
        phi = updateLevelSetFunction(input, phi);

        if(i > 0 && i % reinitialize == 0)
            phi = calculateSignedDistanceTransform(phi);
    }

}

int main(int argc, char ** argv) {

    // Load volume
    Volume<short> * input = new Volume<short>(int3(100, 100, 100));

    // Set initial mask
    int3 origin(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    int size = atoi(argv[5]);

    Volume<float> * initialMask = createInitialMask(origin, size, input->getSize());
    initialMask->show(0.0, 2.0f);
    Volume<float> * test = calculateSignedDistanceTransform(initialMask);
    test->show(0.0f, 20.0f);

    /*
    // Do level set
    runLevelSet(input, initialMask, 100, 20);
    */

    // Visualize result
}
