#include "common.cuh"
#include "raytracing.cuh"
#include "bxdf.cuh"
#include "payload.cuh"
#include "light.cuh"
enum BxdfType {
	Dielectric,
	Metal,
	Glass
};

extern "C" GLOBAL void __raygen__principled_bsdf() {
    // 没有显式递归的版本
    // 主体为一个循环，循环开始时根据上一个循环或rg产生的射线方向进行一次追踪。
    // 计算间接光的权重和直接光的辐射
    const uint3 idx = optixGetLaunchIndex();
    uint pixel_id = idx.y * RayTracingGlobalParams.Width + idx.x;
    const uint3 dim = optixGetLaunchDimensions();

    float3 RayOrigin, RayDirection;
    float2 jitter = ASSERT_VALID(Hammersley(RayTracingGlobalParams.FrameNumber % 32, 32));
    ComputeRayWithJitter(idx, dim, RayOrigin, RayDirection, jitter);

    float3 Weight = ASSERT_VALID(make_float3(1.0f));
    float3 Radiance = ASSERT_VALID(make_float3(0.0f));

    uint RecursionDepth = 0;

    // 首先发射primary ray
    HitInfo hitInfo;
    TraceRay(hitInfo, RayOrigin, RayDirection, TMIN, 0, 1, 0);

    if (hitInfo.surfaceType == Miss) {
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection));
        return;
    }
    else if (hitInfo.surfaceType == Light) {
        ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
        float3 LightColor = ASSERT_VALID(GetColorFromAnyLight(FetchLightData(proceduralMatPtr)));
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = LightColor;
        return;
    }

    float3 DebugColor = ASSERT_VALID(make_float3(0));

    for (; RecursionDepth < RayTracingGlobalParams.MaxRecursionDepth; RecursionDepth++) {
        SurfaceData surfaceData;
        surfaceData.Clear();
        surfaceData.Load(hitInfo);

        bool IsTransmission;
        float Pf_X;
        float3 BsdfIndirect;
        float4 Noise4 = ASSERT_VALID(hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth)));
        float4 Noise14 = ASSERT_VALID(hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth + 0x11932287U)));
        float3 V = ASSERT_VALID(normalize(-RayDirection));

        {
            float3 HForward;
            RayDirection = ASSERT_VALID(SampleBsdf(surfaceData, make_float3(Noise4.x, Noise4.y, Noise4.z), V, IsTransmission, HForward));
            Pf_X = ASSERT_VALID(EvalPdf(surfaceData, V, RayDirection, IsTransmission, HForward));
            BsdfIndirect = ASSERT_VALID(EvalBsdf(surfaceData, V, RayDirection, IsTransmission, HForward));
        }

        TraceRay(hitInfo, surfaceData.Position, RayDirection, TMIN, 0, 1, 0);

        float3 IrradianceDirect = ASSERT_VALID(make_float3(0));
        float3 IrradianceIndirect = ASSERT_VALID(make_float3(0));
        float3 WeightNew = ASSERT_VALID(make_float3(0));
        bool terminateRay = false;
        bool DomeLight = RayTracingGlobalParams.DomeLightBuffer == nullptr;

        if (!IsTransmission && RayTracingGlobalParams.consoleOptions->debugMode == ConsoleDebugMode::MIS) {
            uint LightToSample = (uint)floor(frac(Noise14.z) * RayTracingGlobalParams.LightListLength);
            LightToSample = min(RayTracingGlobalParams.LightListLength - 1, LightToSample);
            float3 LiDirect, BrdfDirect;

            float4 SampleResult = ASSERT_VALID(SampleLight(LightToSample, Noise14.x, Noise14.y, surfaceData.Position));
            float3 SamplePoint = ASSERT_VALID(make_float3(SampleResult.x, SampleResult.y, SampleResult.z));
            float3 RayDirDirectLight = ASSERT_VALID(normalize(SamplePoint - surfaceData.Position));

            HitInfo hitInfoDirectLight;
            TraceRay(hitInfoDirectLight, surfaceData.Position, RayDirDirectLight, TMIN, 0, 1, 0);

            LiDirect = hitInfoDirectLight.surfaceType == SurfaceType::Light
                ? ASSERT_VALID(GetColorFromAnyLight(FetchLightData(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfoDirectLight.SbtDataPtr))))
                : ASSERT_VALID(make_float3(0.0f));

            BrdfDirect = ASSERT_VALID(EvalBsdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight)));

            float Pf_Y = ASSERT_VALID(EvalPdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight)));
            float WeightSum = Pf_Y;

            for (uint light = 0; light < RayTracingGlobalParams.LightListLength; light++) {
                float Pg_Y = ASSERT_VALID(PdfLight(light, surfaceData.Position, RayDirDirectLight) / RayTracingGlobalParams.LightListLength);
                WeightSum += Pg_Y;
            }
            IrradianceDirect += ASSERT_VALID(BrdfDirect * LiDirect / WeightSum);

            WeightSum = Pf_X;
            for (uint light = 0; light < RayTracingGlobalParams.LightListLength; light++) {
                float Pg_X = ASSERT_VALID(PdfLight(light, surfaceData.Position, RayDirection)) / RayTracingGlobalParams.LightListLength;
                WeightSum += Pg_X;
            }

            if (hitInfo.surfaceType == SurfaceType::Miss) {
                IrradianceIndirect = ASSERT_VALID(GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection));
                IrradianceIndirect *= ASSERT_VALID(BsdfIndirect / WeightSum);
                terminateRay = true;
            }
            else if (hitInfo.surfaceType == SurfaceType::Light) {
                float3 LightColor = ASSERT_VALID(GetColorFromAnyLight(FetchLightData(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr))));
                IrradianceIndirect = ASSERT_VALID(LightColor * BsdfIndirect / WeightSum);
                terminateRay = true;
            }
            else {
                WeightNew = ASSERT_VALID(Weight * BsdfIndirect / WeightSum);
            }

            Radiance += ASSERT_VALID(Weight * IrradianceDirect);
            if (terminateRay) {
                Radiance += ASSERT_VALID(Weight * IrradianceIndirect);
            }
        }
        else {
            if (hitInfo.surfaceType == SurfaceType::Miss) {
                IrradianceIndirect = ASSERT_VALID(GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection));
                IrradianceIndirect *= ASSERT_VALID(BsdfIndirect / Pf_X);
                terminateRay = true;
            }
            else if (hitInfo.surfaceType == SurfaceType::Light) {
                IrradianceIndirect = ASSERT_VALID(GetColorFromAnyLight(FetchLightData(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr))));
                IrradianceIndirect *= ASSERT_VALID(BsdfIndirect / Pf_X);
                terminateRay = true;
            }
            else {
                WeightNew = ASSERT_VALID(Weight * BsdfIndirect / Pf_X);
            }

            if (terminateRay) {
                Radiance += ASSERT_VALID(Weight * IrradianceIndirect);
            }
        }

        Weight = WeightNew;
        if (terminateRay) {
            break;
        }
    }

    RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(Radiance);
    return;
}


extern "C" GLOBAL void __closesthit__fetch_hitinfo() {
	HitInfo hitInfo;
	hitInfo.PrimitiveID = optixGetPrimitiveIndex();
	hitInfo.SbtDataPtr = ((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr;
	hitInfo.TriangleCentroidCoord = optixGetTriangleBarycentrics();
	hitInfo.surfaceType = ((ModelData*)hitInfo.SbtDataPtr)->MaterialData->MaterialType == MaterialType::MATERIAL_OBJ ? Opaque : Opaque;
	SetPayLoad(hitInfo);
}

extern "C" GLOBAL void __miss__fetchMissInfo()
{
	HitInfo hitInfo;
	hitInfo.PrimitiveID = 0xFFFFFFFF;
	hitInfo.SbtDataPtr = optixGetSbtDataPointer();
	hitInfo.TriangleCentroidCoord = make_float2(0.0f);
	hitInfo.surfaceType = Miss;
	SetPayLoad(hitInfo);
}


extern "C" GLOBAL void __intersection__sphere_light() {
	float3 ray_origin = optixGetWorldRayOrigin();
	float3 ray_direction = optixGetWorldRayDirection();
	float tmin = optixGetRayTmin();
	float tmax = optixGetRayTmax();
	ProceduralGeometryMaterialBuffer* data = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>();
	float3 pos;
	float radius;
	pos.x = FetchLightData(data)[5];
	pos.y = FetchLightData(data)[6];
	pos.z = FetchLightData(data)[7];
	radius = FetchLightData(data)[8];
	float t = RayIntersectWithSphere(ray_origin, pos, ray_direction, radius, optixGetRayTmin(), optixGetRayTmax());
	if (!isnan(t)) {
		optixReportIntersection(t, 0);
	}
}
extern "C" GLOBAL void __intersection__rectangle_light() {
	float3 ray_origin = optixGetWorldRayOrigin();
	float3 ray_direction = optixGetWorldRayDirection();
	float tmin = optixGetRayTmin();
	float tmax = optixGetRayTmax();
	ProceduralGeometryMaterialBuffer* data = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>();
	float3 p1, p2, p3, p4;
	float3 color;
	RectangleLight::DecodeRectangleLight(data->Elements, p1, p2, p3, p4, color);
	float t = RayIntersectWithRectangle(ray_origin, ray_direction, p1, p2, p3, p4, optixGetRayTmin(), optixGetRayTmax());
	if (!isnan(t)) {
		optixReportIntersection(t, 0);
	}
}
extern "C" GLOBAL void __closesthit__light() {
	HitInfo hitInfo;
	hitInfo.PrimitiveID = 0;
	hitInfo.SbtDataPtr = ((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr;
	hitInfo.TriangleCentroidCoord = make_float2(0, 0);
	hitInfo.surfaceType = SurfaceType::Light;
	SetPayLoad(hitInfo);
}