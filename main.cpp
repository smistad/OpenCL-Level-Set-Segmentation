#include "SIPL/Core.hpp"
#include <cmath>
#include <queue>
#include <iostream>
#include <cfloat>
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
    float threshold = 0.00001f;

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
    phiNext->fill(1000);
#pragma omp parallel for
    for(int z = 1; z < phi->getDepth()-1; z++) {
    for(int y = 1; y < phi->getHeight()-1; y++) {
    for(int x = 1; x < phi->getWidth()-1; x++) {
        int3 pos(x,y,z);

        // Calculate all first order derivatives
        float3 D(
                0.5f*(phi->get(int3(x+1,y,z))-phi->get(int3(x-1,y,z))),
                0.5f*(phi->get(int3(x,y+1,z))-phi->get(int3(x,y-1,z))),
                0.5f*(phi->get(int3(x,y,z+1))-phi->get(int3(x,y,z-1)))
        );
        float3 Dminus(
                phi->get(pos)-phi->get(int3(x-1,y,z)),
                phi->get(pos)-phi->get(int3(x,y-1,z)),
                phi->get(pos)-phi->get(int3(x,y,z-1))
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
                0.5f*(phi->get(int3(x+1,y-1,z))-phi->get(int3(x-1,y-1,z))),
                0.5f*(phi->get(int3(x+1,y,z-1))-phi->get(int3(x-1,y,z-1)))
        );
        float3 DxPlus(
                0.0f,
                0.5f*(phi->get(int3(x+1,y+1,z))-phi->get(int3(x-1,y+1,z))),
                0.5f*(phi->get(int3(x+1,y,z+1))-phi->get(int3(x-1,y,z+1)))
        );
        float3 DyMinus(
                0.5f*(phi->get(int3(x-1,y+1,z))-phi->get(int3(x-1,y-1,z))),
                0.0f,
                0.5f*(phi->get(int3(x,y+1,z-1))-phi->get(int3(x,y-1,z-1)))
        );
        float3 DyPlus(
                0.5f*(phi->get(int3(x+1,y+1,z))-phi->get(int3(x+1,y-1,z))),
                0.0f,
                0.5f*(phi->get(int3(x,y+1,z+1))-phi->get(int3(x,y-1,z+1)))
        );
        float3 DzMinus(
                0.5f*(phi->get(int3(x-1,y,z+1))-phi->get(int3(x-1,y,z-1))),
                0.5f*(phi->get(int3(x,y-1,z+1))-phi->get(int3(x,y-1,z-1))),
                0.0f
        );
        float3 DzPlus(
                0.5f*(phi->get(int3(x+1,y,z+1))-phi->get(int3(x+1,y,z-1))),
                0.5f*(phi->get(int3(x,y+1,z+1))-phi->get(int3(x,y+1,z-1))),
                0.0f
        );

        // Calculate curvature
        float3 nMinus(
                Dminus.x / sqrt(FLT_EPSILON+Dminus.x*Dminus.x+pow(0.5f*(DyMinus.x+D.y),2.0f)+pow(0.5f*(DzMinus.x+D.z),2.0f)),
                Dminus.y / sqrt(FLT_EPSILON+Dminus.y*Dminus.y+pow(0.5f*(DxMinus.y+D.x),2.0f)+pow(0.5f*(DzMinus.y+D.z),2.0f)),
                Dminus.z / sqrt(FLT_EPSILON+Dminus.z*Dminus.z+pow(0.5f*(DxMinus.z+D.x),2.0f)+pow(0.5f*(DyMinus.z+D.y),2.0f))
        );
        float3 nPlus(
                Dplus.x / sqrt(FLT_EPSILON+Dplus.x*Dplus.x+pow(0.5f*(DyPlus.x+D.y),2.0f)+pow(0.5f*(DzPlus.x+D.z),2.0f)),
                Dplus.y / sqrt(FLT_EPSILON+Dplus.y*Dplus.y+pow(0.5f*(DxPlus.y+D.x),2.0f)+pow(0.5f*(DzPlus.y+D.z),2.0f)),
                Dplus.z / sqrt(FLT_EPSILON+Dplus.z*Dplus.z+pow(0.5f*(DxPlus.z+D.x),2.0f)+pow(0.5f*(DyPlus.z+D.y),2.0f))
        );

        float curvature = ((nPlus.x-nMinus.x)+(nPlus.y-nMinus.y)+(nPlus.z-nMinus.z))*0.5;

        // Calculate speed term
        float alpha = 0.03;
        float threshold = 150;
        float epsilon = 100;
        //float speed = -alpha*(epsilon-fabs(input->get(pos)-threshold)) + (1.0f-alpha)*curvature;
        float speed = -alpha*(epsilon-(threshold-input->get(pos))) + (1.0f-alpha)*curvature;

        // Determine gradient based on speed direction
        float3 gradient;
        if(speed < 0) {
            gradient = gradientMin;
        } else {
            gradient = gradientMax;
        }
        if(gradient.length() > 1.0f)
            gradient = gradient.normalize();

        // Stability CFL
        // max(fabs(speed*gradient.length()))
        float deltaT = 0.1f;

        // Update the level set function phi
        phiNext->set(pos, phi->get(pos) + deltaT*speed*gradient.length());
        //std::cout << speed << " " << gradient.length() << std::endl;
    }}}
    delete phi;
    return phiNext;
}

void visualize(Volume<short> * input, Volume<float> * phi) {
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

Volume<float> * runLevelSet(Volume<short> * input, Volume<float> * initialMask, int iterations, int reinitialize) {
    Volume<float> * phi = calculateSignedDistanceTransform(initialMask);
    phi->show(0,255);
    std::cout << "signed distance transform created" << std::endl;

    for(int i = 0; i < iterations; i++) {
        phi = updateLevelSetFunction(input, phi);
        std::cout << "iteration " << (i+1) << " finished " << std::endl;
        //if(i % 10 == 0)
        //visualize(input, phi);
        //phi->show(0,255);

        if(i > 0 && i % reinitialize == 0) {
            phi = calculateSignedDistanceTransform(phi);
            std::cout << "signed distance transform created" << std::endl;
        }
    }

    return phi;
}

void printPhiSlice(Volume<float> * phi) {
   int slice = phi->getDepth() / 2;
   for(int y = 0; y < phi->getHeight(); y++) {
       for(int x = 0; x < phi->getWidth(); x++) {
           std::cout << phi->get(x,y,slice) << "\t";
       }
       std::cout << std::endl;
   }
}


int main(int argc, char ** argv) {

    /*
    Volume<float> * test = createInitialMask(int3(4,4,4), 8, int3(16,16,16));
    Volume<float> * phi = calculateSignedDistanceTransform(test);
    printPhiSlice(phi);
    */

    // Load volume
    Volume<short> * input = new Volume<short>(argv[1]);
    for(int i = 0; i < input->getTotalSize();i++) {
        if(input->get(i) < 0)
            input->set(i,0);
        if(input->get(i) > 200)
            input->set(i,200);
    }

    std::cout << "Dataset of size " << input->getWidth() << ", " << input->getHeight() << ", " << input->getDepth() << std::endl;

    // Set initial mask
    int3 origin(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    int size = atoi(argv[6]);

    Volume<float> * initialMask = createInitialMask(origin, size, input->getSize());
    visualize(input, initialMask);


    // Do level set
    Volume<float> * res = runLevelSet(input, initialMask, atoi(argv[7]), 10000000);

    // Visualize result
    visualize(input, res);

    // Store result
    Volume<char> * segmentation = new Volume<char>(res->getSize());
    segmentation->setSpacing(input->getSpacing());
    for(int i = 0; i < res->getTotalSize(); i++) {
        if(res->get(i) < 0.0f) {
            segmentation->set(i, 1);
        } else {
            segmentation->set(i, 0);
        }
    }

    segmentation->save(argv[2]);
}
