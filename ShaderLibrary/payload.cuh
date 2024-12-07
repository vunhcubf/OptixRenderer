#pragma once
#include "common.cuh"

__device__ void SetPayLoad(HitInfo& payload){
    optixSetPayload_0(payload.PrimitiveID);
    optixSetPayload_1(getLow4Bytes(payload.SbtDataPtr));
    optixSetPayload_2(getHigh4Bytes(payload.SbtDataPtr));
    optixSetPayload_3(__float_as_uint(payload.TriangleCentroidCoord.x));
    optixSetPayload_4(__float_as_uint(payload.TriangleCentroidCoord.y));
    optixSetPayload_5(payload.surfaceType);
}

__device__ void GetPayLoad(HitInfo& payload) {
    payload.PrimitiveID=optixGetPayload_0();
    payload.SbtDataPtr=combineToUint64(optixGetPayload_2(),optixGetPayload_1());
    payload.TriangleCentroidCoord.x=__uint_as_float(optixGetPayload_3());
    payload.TriangleCentroidCoord.y=__uint_as_float(optixGetPayload_4());
    payload.surfaceType=(SurfaceType)optixGetPayload_5();
}

__device__ void TraceRay(
    HitInfo& payload,
    float3 RayOrigin,
    float3 RayDirection,
    float Tmin,
    uint SBTOffset,
    uint SBTStride,
    uint MissSBTIndex) {
    uint p0,p1,p2,p3,p4,p5;
    p0=payload.PrimitiveID;
    p1=getLow4Bytes(payload.SbtDataPtr);
    p2=getHigh4Bytes(payload.SbtDataPtr);
    p3=__float_as_uint(payload.TriangleCentroidCoord.x);
    p4=__float_as_uint(payload.TriangleCentroidCoord.y);
    p5=payload.surfaceType;
    optixTrace(RayTracingGlobalParams.Handle, RayOrigin, RayDirection, Tmin, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        SBTOffset, SBTStride, MissSBTIndex,
        p0,p1,p2,p3,p4,p5);
    payload.PrimitiveID=p0;
    payload.SbtDataPtr=combineToUint64(p2,p1);
    payload.TriangleCentroidCoord.x=__uint_as_float(p3);
    payload.TriangleCentroidCoord.y=__uint_as_float(p4);
    payload.surfaceType=(SurfaceType)p5;
}