#pragma once
#include <sutil/vec_math.h>
#include <optix.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>
#include "optix_device.h"
/// @brief ///////////////////
typedef unsigned int uint;
typedef unsigned long long uint64;
typedef long long int64;
typedef float float32;
typedef double float64;

#define NO_TEXTURE_HERE 0xFFFFFFFF
template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
#define M_PI 3.14159265358979f
#define M_REVERSE_PI 0.318309886183791f
enum MaterialType {
    MATERIAL_AREALIGHT,
    MATERIAL_OBJ
};

struct AreaLight {
    float3 P1, P2, P3, P4, Color;//p1,p2,p3,p4从下面看顺时针一圈
    float Area;
};

struct CameraData {
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
};

struct LaunchParametersDesc {
    CameraData cameraData;
    AreaLight areaLight;
};

struct LaunchParameters {
    float3* IndirectOutputBuffer;
    uchar4* ImagePtr;
    uint Width;
    uint Height;
    CameraData cameraData;
    OptixTraversableHandle Handle;
    AreaLight areaLight;
    uint Seed;
    uint64 FrameNumber;
    uint Spp;
    uint MaxRecursionDepth;
};
struct GeometryBuffer {
    CUdeviceptr Normal=(CUdeviceptr)nullptr;
    CUdeviceptr Vertices = (CUdeviceptr)nullptr;
    CUdeviceptr uv = (CUdeviceptr)nullptr;
};

struct RayGenData
{
    float r, g, b;
    cudaTextureObject_t TestTex;
};
struct MissData {
    float3 BackgroundColor;
    float SkyBoxIntensity;
    cudaTextureObject_t SkyBox;
};

struct SbtDataStruct
{
    CUdeviceptr DataPtr;
};
//原理化BSDF
struct Material
{
    cudaTextureObject_t NormalMap=NO_TEXTURE_HERE;
    cudaTextureObject_t BaseColorMap= NO_TEXTURE_HERE;
    cudaTextureObject_t ARMMap= NO_TEXTURE_HERE;
    float3 BaseColor = make_float3(0.8,0.8,0.8);
    float3 Emission = make_float3(0, 0, 0);
    float Roughness=0.5f;
    float Metallic = 0.0f;
    float Specular=1.f;
    float Transmission = 0.0f;
    float Ior = 1.4f;
    float SpecularTint=0.0f;
    float Opacity = 1.0f;
    MaterialType MaterialType = MaterialType::MATERIAL_OBJ;
};
struct ModelData {
    GeometryBuffer* GeometryData;
    Material* MaterialData;
};
/////////////////////////////////////
//判断光线命中的是灯光还是场景
#define HIT_TYPE_SCENE 0
#define HIT_TYPE_LIGHT 1

#define BXDF_RAY_TYPE_DIFF 1U
#define BXDF_RAY_TYPE_SPEC 2U
#define BXDF_RAY_TYPE_TRANS 4U
extern "C" __constant__ LaunchParameters params;

enum PathTracerRayType {
	RAYTYPE_RAYGEN,
	RAYTYPE_CH_INDIRECT,
	RAYTYPE_CH_OCCLUDED
};

struct PerRayData {
	float3 Radience;
	uint RecursionDepth;
	uint Seed;
	uint RayHitType;
	float3 DebugData;
};
template<typename T>
static __forceinline__ __device__ T SampleTexture2D(cudaTextureObject_t tex, float u, float v) {
	return tex2D<T>(tex, u, v);
}
template<typename T>
static __forceinline__ __device__ float3 SampleTexture2DColor(cudaTextureObject_t tex, float u, float v) {
	float4 tmp= tex2D<T>(tex, u, v);
	float3 color = make_float3(tmp.x, tmp.y, tmp.z);
	return color;
}
static __forceinline__ __device__ float pow2(float a) {
	return a * a;
}
static __forceinline__ __device__ float3 sqrt(float3 a) {
	return make_float3(sqrt(a.x), sqrt(a.y), sqrt(a.z));
}
static __forceinline__ __device__ float Pow4(float a) {
	return a * a * a * a;
}static __forceinline__ __device__ float3 abs(float3 a) {
	return make_float3(abs(a.x), abs(a.y), abs(a.z));
}
static __forceinline__ __device__ float squared_length(float3 vec) {
	return dot(vec, vec);
}
static __forceinline__ __device__ float saturate(float a) {
	return clamp(a, 0.0f, 1.0f);
}
static __forceinline__ __device__ float3 saturate(float3 a) {
	return make_float3(saturate(a.x), saturate(a.y), saturate(a.z));
}
static __forceinline__ __device__ float Pow5(float a) {
	return a * a * a * a * a;
}
static __forceinline__ __device__ float rcp(float a) {
	return 1 / a;
}
static __forceinline__ __device__ float3 min(float3 a, float3 b) {
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

static __forceinline__ __device__ float lerp(float x1, float x2, float t) {
	return x1 * (1 - t) + t * x2;
}
static __forceinline__ __device__ float sign(float x) {
	return x == 0.0f ? 0.0f : (x > 0.0f ? 1.0f : -1.0f);
}
static __device__ __forceinline__ float RadicalInverse_VdC(uint bits)
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
static __device__ __forceinline__ float2 Hammersley(uint i, uint N)
{
	return make_float2((float)i/(float)N, RadicalInverse_VdC(i));
}

static __device__ __forceinline__ float3 FilterGlossy(float3 In,float Threshold) {
	In = fmaxf(make_float3(0.0f), In);
	if (isnan(In.x) || isnan(In.y) || isnan(In.z)) {
		return make_float3(0.0f);
	}
	float MaxValue = fmaxf(In.x, fmaxf(In.y, In.z));
	if (MaxValue > Threshold) {
		return In * Threshold / MaxValue;
	}
	else {
		return In;
	}
}
// ior为折射介质 / 入射介质
__device__ float3 refract(float3 I, float3 N, float3 M, float eta, bool* IsInternalReflection)
{
	if(IsInternalReflection)
		*IsInternalReflection = false;
	float c = dot(I, M);
	float3 L = (eta * c - sign(dot(I, N)) * sqrt(1 + eta * (c * c - 1))) * M - eta * I;
	return L;
}

static __host__ __device__ __inline__ uint lcg3(uint prev)
{
	const uint LCG_A = 1664525u;
	const uint LCG_C = 1013904223u;
	prev = (LCG_A * prev + LCG_C);
	return prev;
}

static __host__ __device__ __inline__ uint lcg4(uint prev)
{
	prev = (prev * 8121 + 28411) % 134456;
	return prev;
}

#define PI 3.14159265358979f
#define REVERSE_PI 0.318309886183791f

static __forceinline__ __device__ float2 GetSkyBoxUv(float3 RayDir) {
	//首先获取垂直方向
	float2 uv;
	uv.y = acos(RayDir.z) / PI;
	uv.y = 1 - uv.y;
	float tan_xy = RayDir.y / RayDir.x;
	if (RayDir.x > 0) {
		uv.x = atan(tan_xy);
	}
	else {
		uv.x = atan(tan_xy)+PI;
	}
	uv.x += PI / 2;
	uv.x /= 2 * PI;
	return uv;
}


static __device__ float Rand(uint& seed) {
	const uint3 id = optixGetLaunchIndex();
	uint seed1 = tea<4>(id.y * params.Width + id.x, seed);
	seed += 0xFC879023U;
	return rnd(seed1);
}
static __device__ float3 RandomSamplePointOnLight(uint& Seed) {
	float r1 = Rand(Seed);
	float r2 = Rand(Seed);
	AreaLight light = params.areaLight;
	return lerp(
		lerp(light.P1, light.P2, r1),
		lerp(light.P4, light.P3, r1),
		r2);
}
static __device__ float3 ImportanceSampleCosWeight(uint& Seed,float3 N) {
	float phi = Rand(Seed);
	float theta = Rand(Seed) * 2.0f * PI;
	phi = asin(sqrt(phi));
	float3 T;
	if (N.x == 0 && N.z == 0) {
		T = make_float3(0, N.z, -N.y);//x
	}
	else {
		T = make_float3(N.z, 0, -N.x);//x
	}
	T = normalize(T);
	float3 B = cross(N, T);//y
	B = normalize(B);
	float3 RayDir = T * sin(phi) * cos(theta) + B * sin(phi) * sin(theta) + N * saturate(cos(phi));
	RayDir = normalize(RayDir);
	return RayDir;
}
static __device__ float3 ImportanceSampleGGX(uint& Seed, float* Pdf, float roughness)
{
	float a = roughness * roughness;
	float2 Xi;
	Xi.x = Rand(Seed);
	Xi.y = Rand(Seed);
	float phi = 2.0 * PI * Xi.x;
	float up = (1.0 - Xi.y) * 1000;
	float down = (1.0 + (a * a - 1.0) * Xi.y) * 1000;
	float cosTheta = sqrt(up / down);
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	// from spherical coordinates to cartesian coordinates
	float3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;

	// 计算pdf_h : D * cos
	if (Pdf) {
		float d = (cosTheta * a - cosTheta) * cosTheta + 1;
		float D = a / (PI * d * d);
		Pdf[0] = D * cosTheta;
	}
	
	return H;
}
static __forceinline__ __device__ void GetTBNFromN(float3 N,float3& T,float3& B) {
	if (N.x == 0 && N.z == 0) {
		T = make_float3(0, N.z, -N.y);//x
	}
	else {
		T = make_float3(N.z, 0, -N.x);//x
	}
	T = normalize(T);
	B = cross(N, T);//y
	B = normalize(B);
}
static __device__ float3 ClmapRayDir(const float3& n, float3 l) {
	float3 T, B, L;
	GetTBNFromN(n, T, B);
	L = make_float3(dot(T, l), dot(B, l), dot(n, l));
	L.z = fmaxf(L.z, 1e-2f);
	L = normalize(L);
	L = T * L.x + B * L.y + n * L.z;
	L = normalize(L);
	return L;
}
static __forceinline__ __device__ float3 UseNormalMap(float3 N,float3 NormalMap,float Intensity) {
	float3 T;
	float3 BT;
	if (abs(N.x) < 1e-4 && abs(N.z) < 1e-4) {
		T = make_float3(0, -N.z, N.y);
	}
	else {
		T = make_float3(N.z, 0, -N.x);
	}
	T = normalize(T);
	BT = cross(T, N);
	NormalMap = NormalMap * 2 - 1;
	float3 N_new = NormalMap.x * T + NormalMap.y * BT + NormalMap.z * N;
	N = lerp(N, N_new, Intensity);
	N = normalize(N);
	return N;
}
static __device__ float3 refract(const float3 incident, const float3 normal, const float eta,bool* internal_reflection)
{
	float k = 1.0f - eta * eta * (1.0f - dot(normal, incident) * dot(normal, incident));
	if (k < 0.0f) {
		if (internal_reflection) {
			internal_reflection[0] = true;
		}
		return make_float3(0);
	}	
	else
		return eta * incident - (eta * dot(normal, incident) + sqrt(k)) * normal;
}