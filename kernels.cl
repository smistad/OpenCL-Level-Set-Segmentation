
__kernel void updateLevelSetFunction(
        __read_only image3d_t input,
        __read_only image3d_t phi_read,
        __write_only image3d_t phi_write
        ) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    const int4 pos = {x,y,z,0};

    // Calculate all first order derivatives
    float3 D(
            0.5f*(read_imagef(phi_read,sampler,int4(x+1,y,z)).x-read_imagef(phi_read,sampler,int4(x-1,y,z)).x),
            0.5f*(read_imagef(phi_read,sampler,int4(x,y+1,z)).x-read_imagef(phi_read,sampler,int4(x,y-1,z)).x),
            0.5f*(read_imagef(phi_read,sampler,int4(x,y,z+1)).x-read_imagef(phi_read,sampler,int4(x,y,z-1)).x)
    );
    float3 Dminus(
            read_imagef(phi_read,sampler,pos).x-read_imagef(phi_read,sampler,int4(x-1,y,z)).x,
            read_imagef(phi_read,sampler,pos).x-read_imagef(phi_read,sampler,int4(x,y-1,z)).x,
            read_imagef(phi_read,sampler,pos).x-read_imagef(phi_read,sampler,int4(x,y,z-1)).x
    );
    float3 Dplus(
            read_imagef(phi_read,sampler,int4(x+1,y,z)).x-read_imagef(phi_read,sampler,pos).x,
            read_imagef(phi_read,sampler,int4(x,y+1,z)).x-read_imagef(phi_read,sampler,pos).x,
            read_imagef(phi_read,sampler,int4(x,y,z+1)).x-read_imagef(phi_read,sampler,pos).x
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
            0.5f*(read_imagef(phi_read,sampler,int4(x+1,y-1,z)).x-read_imagef(phi_read,sampler,int4(x-1,y-1,z)).x),
            0.5f*(read_imagef(phi_read,sampler,int4(x+1,y,z-1)).x-read_imagef(phi_read,sampler,int4(x-1,y,z-1)).x)
    );
    float3 DxPlus(
            0.0f,
            0.5f*(read_imagef(phi_read,sampler,int4(x+1,y+1,z)).x-read_imagef(phi_read,sampler,int4(x-1,y+1,z)).x),
            0.5f*(read_imagef(phi_read,sampler,int4(x+1,y,z+1)).x-read_imagef(phi_read,sampler,int4(x-1,y,z+1)).x)
    );
    float3 DyMinus(
            0.5f*(read_imagef(phi_read,sampler,int4(x-1,y+1,z)).x-read_imagef(phi_read,sampler,int4(x-1,y-1,z)).x),
            0.0f,
            0.5f*(read_imagef(phi_read,sampler,int4(x,y+1,z-1)).x-read_imagef(phi_read,sampler,int4(x,y-1,z-1)).x)
    );
    float3 DyPlus(
            0.5f*(read_imagef(phi_read,sampler,int4(x+1,y+1,z)).x-read_imagef(phi_read,sampler,int4(x+1,y-1,z)).x),
            0.0f,
            0.5f*(read_imagef(phi_read,sampler,int4(x,y+1,z+1)).x-read_imagef(phi_read,sampler,int4(x,y-1,z+1)).x)
    );
    float3 DzMinus(
            0.5f*(read_imagef(phi_read,sampler,int4(x-1,y,z+1)).x-read_imagef(phi_read,sampler,int4(x-1,y,z-1)).x),
            0.5f*(read_imagef(phi_read,sampler,int4(x,y-1,z+1)).x-read_imagef(phi_read,sampler,int4(x,y-1,z-1)).x),
            0.0f
    );
    float3 DzPlus(
            0.5f*(read_imagef(phi_read,sampler,int4(x+1,y,z+1)).x-read_imagef(phi_read,sampler,int4(x+1,y,z-1)).x),
            0.5f*(read_imagef(phi_read,sampler,int4(x,y+1,z+1)).x-read_imagef(phi_read,sampler,int4(x,y+1,z-1)).x),
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

    float curvature = ((nPlus.x-nMinus.x)+(nPlus.y-nMinus.y)+(nPlus.z-nMinus.z))*0.5f;

    // Calculate speed term
    float alpha = 0.02;
    float threshold = 150;
    float epsilon = 50;
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
    phiNext->set(pos, read_imagef(phi_read,sampler,pos) + deltaT*speed*gradient.length());
}
