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
    
    float3 DebugColor = ASSERT_VALID(make_float3(0));

    if (CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::PrimaryRayHitObject) {
        if (hitInfo.surfaceType == Miss) {
            DebugColor = make_float3(1, 0, 1);
        }
        else if (hitInfo.surfaceType == Light) {
            DebugColor = make_float3(0, 1, 0);
        }
        else if (hitInfo.surfaceType == Opaque) {
            DebugColor = make_float3(1, 1, 0);
        }
        else if (hitInfo.surfaceType == ProceduralObject) {
            DebugColor = make_float3(1, 0, 0);
        }
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(DebugColor);
        return;
    }

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
    
    float3 radianceDirect = make_float3(0);
    float3 radianceIndirect = make_float3(0);
    for (; RecursionDepth < RayTracingGlobalParams.MaxRecursionDepth; RecursionDepth++) {
        SurfaceData surfaceData;
        surfaceData.Clear();
        surfaceData.Load(hitInfo);
        // 从一个表面开始
        // X表示Bxdf射线
        // Y表示NEE
        bool IsTransmissionBxdfRay;
        // bxdf射线的bxdf概率
        float Pf_X = 0;
        float Pg_Y = 0;
        float Pf_Y = 0;
        float Pg_X = 0;
        float3 BsdfIndirect;
        float4 Noise4 = ASSERT_VALID(saturate(hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth))));
        float4 Noise14 = ASSERT_VALID(saturate(hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth + 0x11932287U))));
        float4 Noise24 = ASSERT_VALID(saturate(hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth + 0x74308147U))));
        float3 V = ASSERT_VALID(normalize(-RayDirection));
        // 生成bxdf射线
        {
            float3 HForward;
            RayDirection = ASSERT_VALID(SampleBsdf(surfaceData, make_float3(Noise4.x, Noise4.y, Noise4.z), V, IsTransmissionBxdfRay, HForward));
            Pf_X = ASSERT_VALID(EvalPdf(surfaceData, V, RayDirection, IsTransmissionBxdfRay, HForward));
            BsdfIndirect = ASSERT_VALID(EvalBsdf(surfaceData, V, RayDirection, IsTransmissionBxdfRay, HForward));
        }

        TraceRay(hitInfo, surfaceData.Position, RayDirection, TMIN, 0, 1, 0);
        
        if ((RecursionDepth==0 && CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::FirstBsdfRayHitObject) ||
            (RecursionDepth == 1 && CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::SecondBsdfRayHitObject) ||
            (RecursionDepth == 2 && CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::ThirdBsdfRayHitObject)) {
            if (hitInfo.surfaceType == Miss) {
                DebugColor = make_float3(1, 0, 1);
            }
            else if (hitInfo.surfaceType == Light) {
                DebugColor = make_float3(0, 1, 0);
            }
            else if (hitInfo.surfaceType == Opaque) {
                DebugColor = make_float3(1, 1, 0);
            }
            else if (hitInfo.surfaceType == ProceduralObject) {
                DebugColor = make_float3(1, 0, 0);
            }
            RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(DebugColor);
            return;
        }

        
        float3 WeightNew = make_float3(0);
        bool terminateRay = false;
        bool HasDomeLight = RayTracingGlobalParams.DomeLightBuffer != nullptr;

        // 若光线出现在错误的表面则terminateRay
        if (IsTransmissionBxdfRay) {
            terminateRay = !IsRayContributeToBtdf(RayDirection, surfaceData.VertexNormal, V);
        }
        else {
            terminateRay = IsRayContributeToBtdf(RayDirection, surfaceData.VertexNormal, V);
        }
        // 生成nee射线
        bool IsSampleDomeLight = false;
        {
            // 采样灯光,对场景里所有的光源积分
            uint LightToSample = (uint)floor(frac(Noise14.z) * RayTracingGlobalParams.LightListLength + (HasDomeLight ? 1 : 0));
            LightToSample = min(RayTracingGlobalParams.LightListLength - 1 + (HasDomeLight ? 1 : 0), LightToSample);
            float3 LiDirect, BsdfDirect;
            float3 RayDirDirectLight;
            if (LightToSample < RayTracingGlobalParams.LightListLength) {
                float4 SampleResult = ASSERT_VALID(SampleLight(LightToSample, Noise14.x, Noise14.y, surfaceData.Position));
                float3 SamplePoint = ASSERT_VALID(make_float3(SampleResult.x, SampleResult.y, SampleResult.z));
                RayDirDirectLight = ASSERT_VALID(saturateRay(normalize(SamplePoint - surfaceData.Position)));
            }
            else {
                 RayDirDirectLight = ASSERT_VALID(SampleDomeLight(Noise24,make_float2(Noise14.w, Noise4.w)));
                 IsSampleDomeLight = true;
            }
            
            HitInfo hitInfoDirectLight;
            TraceRay(hitInfoDirectLight, surfaceData.Position, RayDirDirectLight, TMIN, 0, 1, 0);

            if ((RecursionDepth == 0 && CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::FirstNEERayHitObject) ||
                (RecursionDepth == 1 && CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::SecondNEERayHitObject) ||
                (RecursionDepth == 2 && CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::ThirdNEERayHitObject)) {
                if (hitInfoDirectLight.surfaceType == Miss) {
                    DebugColor = make_float3(1, 0, 1);
                }
                else if (hitInfoDirectLight.surfaceType == Light) {
                    DebugColor = make_float3(0, 1, 0);
                }
                else if (hitInfoDirectLight.surfaceType == Opaque) {
                    DebugColor = make_float3(1, 1, 0);
                }
                else if (hitInfoDirectLight.surfaceType == ProceduralObject) {
                    DebugColor = make_float3(1, 0, 0);
                }
                RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(DebugColor);
                return;
            }
            // 光源的辐射
            LiDirect = hitInfoDirectLight.surfaceType == SurfaceType::Light
                ? ASSERT_VALID(GetColorFromAnyLight(FetchLightData(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfoDirectLight.SbtDataPtr))))
                : ASSERT_VALID(make_float3(0.0f));
            LiDirect += hitInfoDirectLight.surfaceType == SurfaceType::Miss
                ? ASSERT_VALID(GetSkyBoxColor(hitInfoDirectLight.SbtDataPtr, RayDirDirectLight))
                : ASSERT_VALID(make_float3(0.0f));
            // 判断光源射线是对brdf做贡献，还是对btdf做贡献
            bool NEERayContributeToBtdf = IsRayContributeToBtdf(RayDirection, surfaceData.VertexNormal, V);
            BsdfDirect = ASSERT_VALID(EvalBsdf(surfaceData, V, RayDirDirectLight, NEERayContributeToBtdf, normalize(V + RayDirDirectLight)));
            // 光源射线在bxdf的概率
            Pf_Y = ASSERT_VALID(EvalPdf(surfaceData, V, RayDirDirectLight, NEERayContributeToBtdf, normalize(V + RayDirDirectLight)));
            // 累计光源射线在灯光采样的概率
            for (uint light = 0; light < RayTracingGlobalParams.LightListLength; light++) {
                Pg_Y += ASSERT_VALID(PdfLight(light, surfaceData.Position, RayDirDirectLight) / (RayTracingGlobalParams.LightListLength + (HasDomeLight ? 1 : 0)));
            }
            Pg_Y += HasDomeLight ? ASSERT_VALID(GetDomeLightProb(RayDirDirectLight) / (RayTracingGlobalParams.LightListLength + (HasDomeLight ? 1 : 0))) : 0;
            radianceDirect = (BsdfDirect * LiDirect / (Pg_Y+ Pf_Y)); // 这一项对应着 Li*Bsdf/(PfY+PgY)

            // 累计bxdf射线在灯光采样中的概率Pgx
            for (uint light = 0; light < RayTracingGlobalParams.LightListLength; light++) {
                Pg_X += ASSERT_VALID(PdfLight(light, surfaceData.Position, RayDirection)) / (RayTracingGlobalParams.LightListLength + (HasDomeLight ? 1 : 0));
            }
            Pg_X += HasDomeLight ? ASSERT_VALID(GetDomeLightProb(RayDirection) / (RayTracingGlobalParams.LightListLength + (HasDomeLight ? 1 : 0))) : 0;

            // 现在考虑bxdf射线命中了哪里
            // 若命中光源，统计辐射并结束，若命中物体，累计bxdf射线的bsdf值
            bool BxdfRayHitLightSrc = false;
            float3 LiBxdf=make_float3(0.0f);
            if (hitInfo.surfaceType == SurfaceType::Miss) {
                LiBxdf += ASSERT_VALID(GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection));
                terminateRay = true;
                BxdfRayHitLightSrc = true;
            }
            else if (hitInfo.surfaceType == SurfaceType::Light) {
                LiBxdf += ASSERT_VALID(GetColorFromAnyLight(FetchLightData(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr))));
                terminateRay = true;
                BxdfRayHitLightSrc = true;
            }

            if (BxdfRayHitLightSrc) {
                radianceIndirect = LiBxdf * BsdfIndirect / (Pf_X + Pg_X);
            }
            else {
                ASSERT_VALID(Pf_X);
                ASSERT_VALID(Weight);
                ASSERT_VALID(BsdfIndirect);
                WeightNew = ASSERT_VALID(Weight * BsdfIndirect / (Pf_X + 1e-7f));
            }

            Radiance += ASSERT_VALID(Weight * radianceDirect);
            if (terminateRay) {
                Radiance += ASSERT_VALID(Weight * radianceIndirect);
            }
        }

        Weight = WeightNew;
        if (terminateRay) {
            break;
        }
    }
    if (CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::FinalBsdfRayHitObject) {
        if (hitInfo.surfaceType == Miss) {
            DebugColor = make_float3(1, 0, 1);
        }
        else if (hitInfo.surfaceType == Light) {
            DebugColor = make_float3(0, 1, 0);
        }
        else if (hitInfo.surfaceType == Opaque) {
            DebugColor = make_float3(1, 1, 0);
        }
        else if (hitInfo.surfaceType == ProceduralObject) {
            DebugColor = make_float3(1, 0, 0);
        }
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(DebugColor);
        return;
    }
    if (CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::FinalRadianceDirect) {
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(radianceDirect);
        return;
    }
    if (CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::FinalRadianceIndirect) {
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(radianceIndirect);
        return;
    }
    if (CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::FinalWeight) {
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(Weight);
        return;
    }
    if (CONSOLE_OPTIONS->debugMode == ConsoleDebugMode::FinalWeightClip) {
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(ValueClip(Weight));
        return;
    }
    if (CONSOLE_OPTIONS->debugMode != ConsoleDebugMode::NoDebug) {
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(DebugColor);
    }
    else {
        RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = ASSERT_VALID(Radiance);
    }
    
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