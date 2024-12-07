#pragma once
#include "common.cuh"
#include "curand_kernel.h"

class RandomSequenceGenerator{
private:
    curandStateScrambledSobol64_t* RandomStates;
    curandDirectionVectors64_t* DeviceVectors;
    uint64* ScrambleConstant;
    uint* PixelOffset;
public:
    __device__ float RndUniform(){
        uint3 id=optixGetLaunchIndex();
        uint threadid=id.y*RayTracingGlobalParams.Width+id.x;
        uint64 ThreadCount=RayTracingGlobalParams.Width*RayTracingGlobalParams.Height;
        curand_init(DeviceVectors[0],ScrambleConstant[0],
            PixelOffset[threadid]*ThreadCount+threadid  +  threadid+ThreadCount*RayTracingGlobalParams.FrameNumber*20,
            &RandomStates[threadid]);
        atomicAdd(&PixelOffset[threadid],1);
        return curand_uniform(&RandomStates[threadid]);
    }
};