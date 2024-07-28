#pragma once
#include <sutil/vec_math.h>
#include <optix.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <optix.h>
#include <cuda/random.h>
#include <curand_kernel.h>
#include <cuda/helpers.h>
#include "optix_device.h"
/// @brief ///////////////////
const float goldenRatioConjugate = 0.061803398875f;
typedef unsigned int uint;
typedef unsigned long long uint64;
typedef long long int64;
typedef float float32;
typedef double float64;
enum SurfaceType : uint{
	Light,
	Opaque,
	Miss
};
__device__ __forceinline__ uint3 operator>>(uint3 x,uint i){
	return make_uint3(x.x>>i,x.y>>i,x.z>>i);
}
__device__ __forceinline__ uint3 operator^(uint3 a,uint3 b){
	return make_uint3(a.x^b.x,a.y^b.y,a.z^b.z);
}
__device__ __forceinline__ uint4 operator>>(uint4 x,uint i){
	return make_uint4(x.x>>i,x.y>>i,x.z>>i,x.w>>i);
}
__device__ __forceinline__ uint4 operator^(uint4 a,uint4 b){
	return make_uint4(a.x^b.x,a.y^b.y,a.z^b.z,a.w^b.w);
}
__device__ __forceinline__ uint4 hash44i(uint4 x){
    x = ((x >> 16u) ^ make_uint4(x.y,x.z,x.w,x.x)) * 0x45d9f3bu;
    x = ((x >> 16u) ^ make_uint4(x.y,x.z,x.w,x.x)) * 0x45d9f3bu;
    x = ((x >> 16u) ^ make_uint4(x.y,x.z,x.w,x.x)) * 0x45d9f3bu;
    x = ((x >> 16u) ^ make_uint4(x.y,x.z,x.w,x.x)) * 0x45d9f3bu;
    return x;
}
__device__ __forceinline__ uint4 hash34i(uint3 x0){
    uint4 x = make_uint4(x0.x,x0.y,x0.z,x0.z);
    x = ((x >> 16u) ^ make_uint4(x.y,x.z,x.x,x.y)) * 0x45d9f3bu;
    x = ((x >> 16u) ^ make_uint4(x.y,x.z,x.x,x.z)) * 0x45d9f3bu;
    x = ((x >> 16u) ^ make_uint4(x.y,x.z,x.x,x.x)) * 0x45d9f3bu;
    //x = (x >> 16u) ^ x;
    return x;
}
__device__ __forceinline__ float4 hash44(uint4 p){
    const float scale = pow(2., -32.);
    uint4 h = hash44i(p);
    return make_float4(h)*scale;
}

__device__ __forceinline__ float4 hash34(uint3 p){
    const float scale = 1.0/float(0xffffffffU);
    uint4 h = hash34i(uint3(p));
    return make_float4(h)*scale;
}
__device__ __forceinline__ float3 hash33( uint3 x )
{
	const uint k = 1103515245U;
    x = ((x>>8U)^make_uint3(x.y,x.z,x.x))*k;
    x = ((x>>8U)^make_uint3(x.y,x.z,x.x))*k;
    x = ((x>>8U)^make_uint3(x.y,x.z,x.x))*k;
    
    return make_float3(x)*(1.0/float(0xffffffffU));
}
__device__ __forceinline__ float3 hash33( uint3 x,uint seed )
{
	const uint& k = seed;
    x = ((x>>8U)^make_uint3(x.y,x.z,x.x))*k;
    x = ((x>>8U)^make_uint3(x.y,x.z,x.x))*k;
    x = ((x>>8U)^make_uint3(x.y,x.z,x.x))*k;
    
    return make_float3(x)*(1.0/float(0xffffffffU));
}
__device__ __forceinline__ float frac(float x){
	return x-(int)x;
}
__device__ __forceinline__ float4 frac(float4 a){
	return make_float4(frac(a.x),
	frac(a.y),
	frac(a.z),
	frac(a.w));
}
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
__device__ uint getLow4Bytes(uint64 value) {
    return (uint)(value & 0xFFFFFFFF);
}
__device__ uint64 combineToUint64(uint high4Bytes, uint low4Bytes) {
    return (((uint64)high4Bytes) << 32) | low4Bytes;
}
// 获取 uint64_t 的高 4 字节
__device__ uint getHigh4Bytes(uint64 value) {
    return (uint)((value >> 32) & 0xFFFFFFFF);
}
struct AreaLight {
    float3 P1, P2, P3, P4, Color;//p1,p2,p3,p4从下面看顺时针一圈
    float Area;
};

struct CameraData {
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
};
struct SbtDataStruct
{
    CUdeviceptr DataPtr;
};
struct LaunchParametersDesc {
    CameraData cameraData;
    AreaLight areaLight;
};


struct GeometryBuffer {
    CUdeviceptr Normal=(CUdeviceptr)nullptr;
    CUdeviceptr Vertices = (CUdeviceptr)nullptr;
    CUdeviceptr uv = (CUdeviceptr)nullptr;
};
struct HitInfo{ // 5 uint
	CUdeviceptr SbtDataPtr;// 2 uint
	uint PrimitiveID; // 1 uint
	float2 TriangleCentroidCoord; // 2 uint
	SurfaceType surfaceType;
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
struct BlueNoiseMapBuffer{
    unsigned char* Data;
    int width;
    int height;
    int channel;
	template<uint channels>
	__device__ void Sample(uint2 pixel_id,void* result);
	template<>
	__device__ void Sample<1>(uint2 pixel_id,void* result){
		pixel_id.y=pixel_id.y%height;
		pixel_id.x=pixel_id.x%width;
		uint address_offset=(pixel_id.y * width + pixel_id.x) * channel;
		float* res=(float*)result;
		res[0]=(Data[address_offset]/255.0f);
	}
	template<>
	__device__ void Sample<2>(uint2 pixel_id,void* result){
		pixel_id.y=pixel_id.y%height;
		pixel_id.x=pixel_id.x%width;
		uint address_offset=(pixel_id.y * width + pixel_id.x) * channel;
		float2* res=(float2*)result;
		res->x=(Data[address_offset]/255.0f);
		res->y=(Data[address_offset+1]/255.0f);
	}
	template<>
	__device__ void Sample<3>(uint2 pixel_id,void* result){
		pixel_id.y=pixel_id.y%height;
		pixel_id.x=pixel_id.x%width;
		uint address_offset=(pixel_id.y * width + pixel_id.x) * channel;
		float3* res=(float3*)result;
		res->x=(Data[address_offset]/255.0f);
		res->y=(Data[address_offset+1]/255.0f);
		res->z=(Data[address_offset+2]/255.0f);
	}
	template<>
	__device__ void Sample<4>(uint2 pixel_id,void* result){
		pixel_id.y=pixel_id.y%height;
		pixel_id.x=pixel_id.x%width;
		uint address_offset=(pixel_id.y * width + pixel_id.x) * channel;
		float4* res=(float4*)result;
		res->x=(Data[address_offset]/255.0f);
		res->y=(Data[address_offset+1]/255.0f);
		res->z=(Data[address_offset+2]/255.0f);
		res->w=(Data[address_offset+3]/255.0f);
	}
	__device__ inline uint2 GetLoopSamplePixelId();
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
    // 随机数生成
    uint64* PixelOffset;
	BlueNoiseMapBuffer* BlueNoiseBuffer;
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
extern "C" __constant__ LaunchParameters RayTracingGlobalParams;

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
__device__ inline uint2 BlueNoiseMapBuffer::GetLoopSamplePixelId(){
	uint3 id=optixGetLaunchIndex();
	uint threadid=id.y*RayTracingGlobalParams.Width+id.x;
	uint64 pixeloffset=RayTracingGlobalParams.PixelOffset[threadid];
	atomicAdd(&RayTracingGlobalParams.PixelOffset[threadid],1);
	pixeloffset=pixeloffset%(width*height);
	uint2 res=make_uint2(pixeloffset%width,pixeloffset/width);
	res.x+=id.x;
	res.y+=id.y;
	res.y=res.y%height;
	res.x=res.x%width;
	return res;
}
#define SAMPLE_BLUENOISE_4D(x) RayTracingGlobalParams.BlueNoiseBuffer->Sample<4>(make_uint2(optixGetLaunchIndex().x,optixGetLaunchIndex().y),&x)
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
	uint seed1 = tea<4>(id.y * RayTracingGlobalParams.Width + id.x, seed);
	seed += 0xFC879023U;
	return rnd(seed1);
}
static __device__ float3 RandomSamplePointOnLight(uint& Seed) {
	float r1 = Rand(Seed);
	float r2 = Rand(Seed);
	AreaLight light = RayTracingGlobalParams.areaLight;
	return lerp(
		lerp(light.P1, light.P2, r1),
		lerp(light.P4, light.P3, r1),
		r2);
}
static __device__ float3 RandomSamplePointOnLight(float2 rand) {
	float r1 = rand.x;
	float r2 = rand.y;
	AreaLight light = RayTracingGlobalParams.areaLight;
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
static __device__ float3 ImportanceSampleCosWeight(float2 rand,float3 N) {
	float phi = rand.x;
	float theta = rand.y * 2.0f * PI;
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
static __device__ float3 ImportanceSampleGGX(float2 rnd, float* Pdf, float roughness)
{
	float a = roughness * roughness;
	float2& Xi=rnd;
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
static __device__ void AccumulatePixelOffset(){
	uint3 id=optixGetLaunchIndex();
    uint threadid=id.y*RayTracingGlobalParams.Width+id.x;
	atomicAdd(&RayTracingGlobalParams.PixelOffset[threadid],1);
}
static __device__ uint64 FetchPixelOffset(){
	uint3 id=optixGetLaunchIndex();
    uint threadid=id.y*RayTracingGlobalParams.Width+id.x;
	return RayTracingGlobalParams.PixelOffset[threadid];
}
static __device__ float RndUniform(){
	uint3 id=optixGetLaunchIndex();
	curandStateXORWOW_t state;
	uint threadid=id.y*RayTracingGlobalParams.Width+id.x;
	uint64 ThreadCount=RayTracingGlobalParams.Width*RayTracingGlobalParams.Height;
	curand_init(RayTracingGlobalParams.Seed,
		RayTracingGlobalParams.FrameNumber*100+RayTracingGlobalParams.PixelOffset[threadid],
		threadid,&state);
	atomicAdd(&RayTracingGlobalParams.PixelOffset[threadid],1);
	return curand_uniform(&state);
}

struct SurfaceData{
	float3 Normal;
	float3 GeometryNormal;
	float3 Position;
	float3 BaseColor;
	float3 NormalMap;
	float2 TexCoord;
	float Roughness;
	float Metallic;
	float Specular;
	float SpecularTint;
	float AO=1.0f;
	float Transmission;
	float ior;
	SurfaceType HitType;
	__device__ void Clear(){
		Normal=make_float3(0.0f);
		GeometryNormal=make_float3(0.0f);
		Position=make_float3(0.0f);
		BaseColor=make_float3(0.0f);
		NormalMap=make_float3(0.0f);
		TexCoord=make_float2(0.0f);
		Roughness=0.0f;
		Metallic=0.0f;
		Specular=0.0f;
		SpecularTint=0.0f;
		AO=1.0f;
		Transmission=0.0f;
		ior=0.0f;
	}
	__device__ void Load(HitInfo& hitInfo){
		HitType=hitInfo.surfaceType;
		if(HitType!=SurfaceType::Opaque){
			return;
		}
		if(hitInfo.PrimitiveID==0xFFFFFFFF){return;}
		ModelData* ModelDataptr = (ModelData*)(hitInfo.SbtDataPtr);
		Specular = ModelDataptr->MaterialData->Specular;
		SpecularTint = ModelDataptr->MaterialData->SpecularTint;
		GeometryBuffer* GeometryDataPtr=ModelDataptr->GeometryData;

		float2* UvPtr = (float2*)GeometryDataPtr->uv;
		const uint& primIndex = hitInfo.PrimitiveID;
		float3 Centrics = make_float3(
			1 - hitInfo.TriangleCentroidCoord.x - hitInfo.TriangleCentroidCoord.y, 
			hitInfo.TriangleCentroidCoord.x, 
			hitInfo.TriangleCentroidCoord.y);

		TexCoord= UvPtr[3 * primIndex] * Centrics.x + UvPtr[3 * primIndex+1] * Centrics.y + UvPtr[3 * primIndex+2] * Centrics.z;

		float3* NormalPtr = (float3*)GeometryDataPtr->Normal;
		float3& Normal1 = NormalPtr[3*primIndex];
		float3& Normal2 = NormalPtr[3*primIndex+1];
		float3& Normal3 = NormalPtr[3*primIndex+2];
		Normal= normalize(Normal1 * Centrics.x + Normal2 * Centrics.y + Normal3 * Centrics.z);
	
		// 计算几何法线
		float3* VerticesPtr = (float3*)GeometryDataPtr->Vertices;
		float3& v1 = VerticesPtr[3 * primIndex];
		float3& v2 = VerticesPtr[3 * primIndex + 1];
		float3& v3 = VerticesPtr[3 * primIndex + 2];
		GeometryNormal = normalize(cross(v1 - v2, v1 - v3));
		Position=v1 * Centrics.x + v2 * Centrics.y + v3 * Centrics.z;
		if (ModelDataptr->MaterialData->BaseColorMap != NO_TEXTURE_HERE) {
			float4 tmp = SampleTexture2D<float4>(ModelDataptr->MaterialData->BaseColorMap, TexCoord.x, TexCoord.y);
			BaseColor = make_float3(tmp.x, tmp.y, tmp.z);
		}
		else {
			BaseColor = ModelDataptr->MaterialData->BaseColor;
		}
		
		if (ModelDataptr->MaterialData->ARMMap != NO_TEXTURE_HERE) {
			float4 tmp = SampleTexture2D<float4>(ModelDataptr->MaterialData->ARMMap, TexCoord.x, TexCoord.y);
			Roughness = tmp.y;
			Metallic = tmp.z;
			AO = tmp.x;
		}
		else {
			Roughness = ModelDataptr->MaterialData->Roughness;
			Metallic = ModelDataptr->MaterialData->Metallic;
		}
		BaseColor *= AO;
		Roughness = fmaxf(Roughness, 1e-3f);
		Transmission = ModelDataptr->MaterialData->Transmission;
		ior = ModelDataptr->MaterialData->Ior;
		ior = fmaxf(ior, 1.0001f);
		// 应用法线贴图
		if (ModelDataptr->MaterialData->NormalMap != NO_TEXTURE_HERE) {
			float4 tmp = SampleTexture2D<float4>(ModelDataptr->MaterialData->NormalMap, TexCoord.x, TexCoord.y);
			NormalMap = make_float3(tmp.x, tmp.y, tmp.z);
			Normal = UseNormalMap(Normal, NormalMap, 1.0f);
		}
	}
};