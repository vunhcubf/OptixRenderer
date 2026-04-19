#pragma once
#include "common.cuh"

DEVICE INLINE float* FetchLightData(uint index) {
    CUdeviceptr* LightArrayPtr = (CUdeviceptr*)RayTracingGlobalParams.LightListArrayptr;
    ProceduralGeometryMaterialBuffer* lightDataPtr = (ProceduralGeometryMaterialBuffer*)LightArrayPtr[index];
    Assert(index < RayTracingGlobalParams.LightListLength);
    return lightDataPtr->Elements;
}
DEVICE INLINE float* FetchLightData(ProceduralGeometryMaterialBuffer* ptr) {
    return ptr->Elements;
}

DEVICE INLINE LightType FetchLightType(uint index){
    float* lightData = FetchLightData(index);
    return (LightType)FloatAsUint(lightData[1]);
}

DEVICE INLINE LightType FetchLightType(ProceduralGeometryMaterialBuffer* ptr) {
    float* lightData = ptr->Elements;
    return (LightType)FloatAsUint(lightData[1]);
}
DEVICE INLINE uint FetchLightIndex(ProceduralGeometryMaterialBuffer* ptr) {
    float* lightData = ptr->Elements;
    return FloatAsUint(lightData[0]);
}
DEVICE INLINE float3 GetColorFromAnyLight(float* dataPtr) {
    float3 Color;
    Color.x = dataPtr[2];
    Color.y = dataPtr[3];
    Color.z = dataPtr[4];
    return Color;
}
DEVICE INLINE float3 GetColorFromAnyLight(uint lightIndex) {
    return GetColorFromAnyLight(FetchLightData(lightIndex));
}

class SphereLight {
public:
    DEVICE INLINE static void DecodeSphereLight(float* dataPtr, float3& Pos, float& r, float3& Color) {
        Color.x = dataPtr[2];
        Color.y = dataPtr[3];
        Color.z = dataPtr[4];
        Pos.x = dataPtr[5];
        Pos.y = dataPtr[6];
        Pos.z = dataPtr[7];
        r = dataPtr[8];
    }
    DEVICE INLINE  static float4 SampleSphereLight(float random1, float random2, float3 shading_point, float3 light_center, float radius) {
        float dc = length(light_center - shading_point);
        float sin_theta_max = radius / dc;
        float cos_theta_max = ASSERT_VALID(sqrtf(1.0f - sin_theta_max * sin_theta_max));
        float cos_theta = (1 - random1) + random1 * cos_theta_max;
        cos_theta = saturate(cos_theta);
        float phi = 2 * PI * random2;
        float ds = dc * cos_theta - ASSERT_VALID(sqrtf(saturate(radius * radius - dc * dc * (1 - cos_theta * cos_theta)) ));
        float cos_alpha = ASSERT_VALID((dc * dc + radius * radius - ds * ds) / (2 * dc * radius));

        float3 nNor = normalize(shading_point - light_center);
        float3 nTan;
        if (abs(nNor.y) >= 1e-3f) {
            nTan = normalize(make_float3(0, nNor.z, -nNor.y));
        }
        else {
            nTan = normalize(make_float3(nNor.z, 0, -nNor.x));
        }
        float3 nBiTan = cross(nTan, nNor);

        float sin_alpha = ASSERT_VALID(sqrtf(saturate(1.0f - cos_alpha * cos_alpha)));
        float3 nObj = ASSERT_VALID(make_float3(sin_alpha * cos(phi), sin_alpha * sin(phi), cos_alpha));
        nObj = ASSERT_VALID(nObj.x * nBiTan + nObj.y * nTan + nObj.z * nNor);
        float3 SamplePoint = ASSERT_VALID(normalize(nObj) * radius + light_center);
        float pdf = 1 / (2 * PI * (1 - cos_theta_max));
        return make_float4(SamplePoint, pdf);
    }
    DEVICE  static float PdfSphereLight(float3 shading_point, float3 light_center, float radius, float3 RayDir) {
        float dc = length(light_center - shading_point);
        float sin_theta_max = radius / dc;
        float cos_theta_max = sqrtf(1.0f - sin_theta_max * sin_theta_max);
        if (dot(RayDir, normalize(light_center - shading_point)) < cos_theta_max-1e-4f) {
            return 0.0f;
        }
        float pdf = 1 / (2 * PI * (1.0f - cos_theta_max+1e-7f));
        return pdf;
    }
    DEVICE static float4 SampleAndGetPdf(float* dataPtr,float r1,float r2,float3 shading_point) {
        float3 color, pos;
        float r;
        DecodeSphereLight(dataPtr, pos, r, color);
        return SampleSphereLight(r1, r2, shading_point, pos, r);
    }
    DEVICE static float Pdf(float* dataPtr, float3 shadingPoint, float3 RayDir) {
        float3 color, pos;
        float r;
        DecodeSphereLight(dataPtr, pos, r, color);
        return PdfSphereLight(shadingPoint, pos, r, RayDir);
    }
};

class RectangleLight {
public:
    DEVICE INLINE static void DecodeRectangleLight(float* dataPtr, float3& p1, float3& p2, float3& p3, float3& p4, float3& Color) {
        Color.x = dataPtr[2];
        Color.y = dataPtr[3];
        Color.z = dataPtr[4];
        p1.x = dataPtr[5];
        p1.y = dataPtr[6];
        p1.z = dataPtr[7];
        p2.x = dataPtr[8];
        p2.y = dataPtr[9];
        p2.z = dataPtr[10];
        p3.x = dataPtr[11];
        p3.y = dataPtr[12];
        p3.z = dataPtr[13];
        p4 = p2 + p3 - p1;
    }

    DEVICE INLINE static  float3 SampleRectangleLight(float random1, float random2, float3 p1,float3 p2,float3 p3,float3 p4) {
        float3 b1 = p3 - p1;
        float3 b2 = p2 - p1;
        return b1 * random1 + b2 * random2 + p1;
    }
    DEVICE  static float PdfRectangleLight(float3 shading_point, float3 p1, float3 p2, float3 p3, float3 p4,float3 sample_point) {
        float area = length(p3 - p1) * length(p2 - p1);
        float r2 = dot(sample_point - shading_point, sample_point - shading_point);
        float3 normal = normalize(cross(p3 - p1, p2 - p1));
        float3 raydir = normalize(sample_point - shading_point);
        float cos_theta = abs(dot(raydir, normal));
        return r2 / (area * cos_theta+FloatEpsilon);
    }
    DEVICE static float4 SampleAndGetPdf(float* dataPtr, float r1, float r2, float3 shading_point) {
        float3 color, p1, p2, p3, p4;
        DecodeRectangleLight(dataPtr, p1, p2, p3, p4, color);
        float3 samplePoint = SampleRectangleLight(r1, r2, p1, p2, p3, p4);
        float pdf = PdfRectangleLight(shading_point, p1, p2, p3, p4, samplePoint);
        return make_float4(samplePoint.x, samplePoint.y, samplePoint.z, pdf);
    }
    DEVICE static float Pdf(float* dataPtr, float3 shadingPoint, float3 samplePoint) {
        float3 color, p1, p2, p3, p4;
        DecodeRectangleLight(dataPtr, p1, p2, p3, p4, color);
        float pdf = PdfRectangleLight(shadingPoint, p1, p2, p3, p4, samplePoint);
        return pdf;
    }
};

DEVICE float RayIntersectWithSphere(float3 ray_origin, float3 sphere_center, float3 ray_direction, float sphere_radius, float tmin, float tmax) {
    float3 oc = ray_origin - sphere_center;
    float A = dot(ray_direction, ray_direction);
    float B = 2.0f * dot(oc, ray_direction);
    float C = dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = B * B - 4.0f * A * C;
    if (discriminant > 0.0f) {
        float sqrt_discriminant = sqrtf(discriminant);
        float t1 = (-B - sqrt_discriminant) / (2.0f * A);
        float t2 = (-B + sqrt_discriminant) / (2.0f * A);
        float t = tmax;
        if (t1 > tmin && t1 < tmax) {
            t = t1;
        }
        if (t2 > tmin && t2 < tmax && t2 < t) {
            t = t2;
        }
        return t;
    }
    else {
        return FLOAT_NAN;
    }
}

DEVICE float RayIntersectWithRectangle(float3 ray_origin, float3 ray_direction, float3 p1, float3 p2, float3 p3, float3 p4, float tmin, float tmax) {
    float3 n = normalize(cross(p3 - p1, p2 - p1));
    if (abs(dot(n, ray_direction)) < 1e-9f) {
        return FLOAT_NAN;
    }
    float t = -dot(n, ray_origin - p1) / dot(n, ray_direction);
    if (t > tmax || t < tmin) {
        return FLOAT_NAN;
    }
    float3 hitPoint = ray_origin + ray_direction * t;
    float3 p1a = hitPoint - p1;
    float3 p1p2 = p2 - p1;
    float3 p1p3 = p3 - p1;
    float bx = dot(p1p2, p1a) / dot(p1p2, p1p2);
    float by = dot(p1p3, p1a) / dot(p1p3, p1p3);
    if (bx <= 1.0f+1e-4f && by <= 1.0f + 1e-4f && bx >= 0.0f - 1e-4f && by >= 0.0f - 1e-4f) {
        return t;
    }
    else {
        return FLOAT_NAN;
    }
}
// 通过灯光编号来采样灯光
DEVICE float4 SampleLight(uint lightIndex, float r1, float r2, float3 shadingPoint) {
    float* dataPtr = FetchLightData(lightIndex);
    LightType lightType = FetchLightType(lightIndex);
    if (lightType == LightType::Sphere) {
        return ASSERT_VALID(SphereLight::SampleAndGetPdf(dataPtr, r1, r2, shadingPoint));
    }
    else if (lightType == LightType::Rectangle) {
        return ASSERT_VALID(RectangleLight::SampleAndGetPdf(dataPtr, r1, r2, shadingPoint));
    }
}
DEVICE float PdfLight(uint lightIndex, float3 shadingPoint, float3 rayDir) {
    float* dataPtr = FetchLightData(lightIndex);
    LightType lightType = FetchLightType(lightIndex);
    if (lightType == LightType::Sphere) {
        return SphereLight::Pdf(dataPtr,shadingPoint, rayDir);
    }
    else if (lightType == LightType::Rectangle) {
        float3 p1, p2, p3, p4;
        float3 color;
        RectangleLight::DecodeRectangleLight(dataPtr, p1, p2, p3, p4, color);
        float t = RayIntersectWithRectangle(shadingPoint, rayDir, p1, p2, p3, p4, TMIN, TMAX);
        if (isnan(t)) {
            return 0;
        }
        float3 samplePoint;
        samplePoint = shadingPoint + t * rayDir;
        return RectangleLight::Pdf(dataPtr, shadingPoint, samplePoint);
    }
    else {
        Assert(false);
    }
}

// 关于dome light的代码
DEVICE float3 SampleDomeLight(float4 rnd) {
    DomeLightISStruct* domeLightBuffer = (DomeLightISStruct*)RayTracingGlobalParams.DomeLightBuffer;
    DomeLightISStruct dome = domeLightBuffer[0];
    // ---------- sample row ----------
    int rowIdx = (int)(rnd.x * dome.height);
    if (rowIdx >= dome.height) rowIdx = dome.height - 1;
    if (rowIdx < 0) rowIdx = 0;

    int row;
    if (rnd.y < dome.colQ[rowIdx])
        row = rowIdx;
    else
        row = dome.colAlias[rowIdx];
    // ---------- sample col ----------
    int colIdx = (int)(rnd.z * dome.width);
    if (colIdx >= dome.width) colIdx = dome.width - 1;
    if (colIdx < 0) colIdx = 0;

    int offset = row * dome.width + colIdx;

    int col;
    if (rnd.w < dome.rowQ[offset])
        col = colIdx;
    else
        col = dome.rowAlias[offset];
    // ---------- texel center -> uv ----------
    float u = ((float)col + 0.5f) / (float)dome.width;
    float v = ((float)row + 0.5f) / (float)dome.height;

    float3 rayDir= GetRayDirFromSkyBoxUv(make_float2(u,v));
    return rayDir;
}
DEVICE float GetDomeLightProb(float3 RayDir) {
    DomeLightISStruct* domeLightBuffer =
        (DomeLightISStruct*)RayTracingGlobalParams.DomeLightBuffer;
    DomeLightISStruct dome = domeLightBuffer[0];

    float2 uv = GetSkyBoxUv(RayDir);
    int col = (int)(uv.x * dome.width);
    int row = (int)(uv.y * dome.height);
    if (col >= dome.width)  col = dome.width - 1;
    if (col < 0)            col = 0;
    if (row >= dome.height) row = dome.height - 1;
    if (row < 0)            row = 0;

    float pRow = dome.pdfMarginal[row];
    float pColGivenRow = dome.pdfRow[row * dome.width + col];
    float pTexel = pRow * pColGivenRow;
    if (pTexel <= 0.0f)
        return 0.0f;

    float vCenter = ((float)row + 0.5f) / (float)dome.height;
    float theta = PI * (1.0f - vCenter);

    float sinTheta = sinf(theta);
    if (sinTheta <= 1e-8f)
        return 0.0f;

    float deltaTheta = PI / (float)dome.height;
    float deltaPhi = 2.0f * PI / (float)dome.width;
    float solidAngle = sinTheta * deltaTheta * deltaPhi;

    if (!(solidAngle > 0.0f))
        return 0.0f;

    // 6. 返回对立体角 pdf
    return pTexel / solidAngle;
}