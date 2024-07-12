#include "common.cuh"

static __device__ __forceinline__ PerRayData FetchPerRayDataFromPayLoad() {
	PerRayData data;
	data.Radience.x = __uint_as_float(optixGetPayload_0());
	data.Radience.y = __uint_as_float(optixGetPayload_1());
	data.Radience.z = __uint_as_float(optixGetPayload_2());

	data.RecursionDepth = optixGetPayload_3();
	data.Seed = optixGetPayload_4();
	data.RayHitType = optixGetPayload_5();

	data.DebugData.x = __uint_as_float(optixGetPayload_6());
	data.DebugData.y = __uint_as_float(optixGetPayload_7());
	data.DebugData.z = __uint_as_float(optixGetPayload_8());
	return data;
}
static __device__ __forceinline__ void SetPerRayDataForPayLoad(PerRayData data) {
	optixSetPayload_0(__float_as_uint(data.Radience.x));
	optixSetPayload_1(__float_as_uint(data.Radience.y));
	optixSetPayload_2(__float_as_uint(data.Radience.z));

	optixSetPayload_3(data.RecursionDepth);
	optixSetPayload_4(data.Seed);
	optixSetPayload_5(data.RayHitType);

	optixSetPayload_6(__float_as_uint(data.DebugData.x));
	optixSetPayload_7(__float_as_uint(data.DebugData.y));
	optixSetPayload_8(__float_as_uint(data.DebugData.z));
}
static __device__ __forceinline__ ModelData* GetModelDataPtr() {
	SbtDataStruct* SbtDataStructPtr = (SbtDataStruct*)optixGetSbtDataPointer();
	return (ModelData*)SbtDataStructPtr->DataPtr;
}
static __device__ __forceinline__ Material* GetMaterialDataPtr() {
	ModelData* DataPtr = GetModelDataPtr();
	return DataPtr->MaterialData;
}
static __device__ __forceinline__ GeometryBuffer* GetGeometryDataPtr() {
	ModelData* DataPtr = GetModelDataPtr();
	return DataPtr->GeometryData;
}
static __device__ __forceinline__  float3 GetBaryCentrics() {
	const float2 b = optixGetTriangleBarycentrics();//x是2号顶点的坐标，y是三号顶点的坐标，不知道为什么这么设计
	return make_float3(1 - b.x - b.y, b.x, b.y);
}
static __device__ __forceinline__ float3 GetNormal() {
	GeometryBuffer* GeometryDataPtr = GetGeometryDataPtr();
	float3* NormalPtr = (float3*)GeometryDataPtr->Normal;
	const uint primIndex = optixGetPrimitiveIndex();
	float3 Centrics = GetBaryCentrics();
	float3 Normal1 = NormalPtr[3 * primIndex];
	float3 Normal2 = NormalPtr[3 * primIndex + 1];
	float3 Normal3 = NormalPtr[3 * primIndex + 2];
	return normalize(Normal1 * Centrics.x + Normal2 * Centrics.y + Normal3 * Centrics.z);
}
static __device__ __forceinline__ float3 GetPosition() {
	GeometryBuffer* GeometryDataPtr = GetGeometryDataPtr();
	float3* VerticesPtr = (float3*)GeometryDataPtr->Vertices;
	const uint primIndex = optixGetPrimitiveIndex();
	float3 Centrics = GetBaryCentrics();
	float3 Vertices1 = VerticesPtr[3 * primIndex];
	float3 Vertices2 = VerticesPtr[3 * primIndex + 1];
	float3 Vertices3 = VerticesPtr[3 * primIndex + 2];
	return Vertices1 * Centrics.x + Vertices2 * Centrics.y + Vertices3 * Centrics.z;
}
static __device__ __forceinline__ void GetTriangle(float3& v1,float3& v2,float3& v3) {
	GeometryBuffer* GeometryDataPtr = GetGeometryDataPtr();
	float3* VerticesPtr = (float3*)GeometryDataPtr->Vertices;
	const uint primIndex = optixGetPrimitiveIndex();
	float3 Centrics = GetBaryCentrics();
	v1 = VerticesPtr[3 * primIndex];
	v2 = VerticesPtr[3 * primIndex + 1];
	v3 = VerticesPtr[3 * primIndex + 2];
}
static __device__ __forceinline__ float2 GetTexCoord() {
	GeometryBuffer* GeometryDataPtr = GetGeometryDataPtr();
	float2* UvPtr = (float2*)GeometryDataPtr->uv;
	const uint primIndex = optixGetPrimitiveIndex();
	float3 Centrics = GetBaryCentrics();
	float2 Uv1 = UvPtr[3 * primIndex];
	float2 Uv2 = UvPtr[3 * primIndex + 1];
	float2 Uv3 = UvPtr[3 * primIndex + 2];
	return Uv1 * Centrics.x + Uv2 * Centrics.y + Uv3 * Centrics.z;
}
static __forceinline__ __device__ void ComputeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
	const float3 U = params.cameraData.cam_u;
	const float3 V = params.cameraData.cam_v;
	const float3 W = params.cameraData.cam_w;
	const float2 d = 2.0f * make_float2(
		static_cast<float>(idx.x) / static_cast<float>(dim.x),
		static_cast<float>(idx.y) / static_cast<float>(dim.y)
	) - 1.0f;

	origin = params.cameraData.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}
static __forceinline__ __device__ void ComputeRayWithJitter(uint3 idx, uint3 dim, float3& origin, float3& direction, float2 jitter)
{
	float2 screen_uv = make_float2(idx.x + jitter.x, idx.y + jitter.y);
	screen_uv.x /= dim.x;
	screen_uv.y /= dim.y;
	const float3 U = params.cameraData.cam_u;
	const float3 V = params.cameraData.cam_v;
	const float3 W = params.cameraData.cam_w;
	const float2 d = 2.0f * screen_uv - 1.0f;

	origin = params.cameraData.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}
static __forceinline__ __device__ void optixTraceWithPerRayData(
	PerRayData& data,
	float3 RayOrigin,
	float3 RayDirection,
	float Tmin,
	uint SBTOffset,
	uint SBTStride,
	uint MissSBTIndex) {
	uint p0, p1, p2, p3, p4, p5, p6,p7,p8;
	p0 = __float_as_uint(data.Radience.x);
	p1 = __float_as_uint(data.Radience.y);
	p2 = __float_as_uint(data.Radience.z);
	p3 = data.RecursionDepth;
	p4 = data.Seed;
	p5 = data.RayHitType;
	p6 = __float_as_uint(data.DebugData.x);
	p7 = __float_as_uint(data.DebugData.y);
	p8 = __float_as_uint(data.DebugData.z);
	optixTrace(params.Handle, RayOrigin, RayDirection, Tmin, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		SBTOffset, SBTStride, MissSBTIndex,
		p0, p1, p2, p3, p4, p5,p6,p7,p8);
	//optixReorder();
	//optixInvoke(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);
	data.Radience.x = __uint_as_float(p0);
	data.Radience.y = __uint_as_float(p1);
	data.Radience.z = __uint_as_float(p2);
	data.RecursionDepth = p3;
	data.Seed = p4;
	data.RayHitType = p5;
	data.DebugData.x = __uint_as_float(p6);
	data.DebugData.y = __uint_as_float(p7);
	data.DebugData.z = __uint_as_float(p8);
}
static __forceinline__ __device__ void optixTraceWithPerRayDataReordered(
	PerRayData& data,
	float3 RayOrigin,
	float3 RayDirection,
	float Tmin,
	uint SBTOffset,
	uint SBTStride,
	uint MissSBTIndex) {
	uint p0, p1, p2, p3, p4, p5, p6, p7, p8;
	p0 = __float_as_uint(data.Radience.x);
	p1 = __float_as_uint(data.Radience.y);
	p2 = __float_as_uint(data.Radience.z);
	p3 = data.RecursionDepth;
	p4 = data.Seed;
	p5 = data.RayHitType;
	p6 = __float_as_uint(data.DebugData.x);
	p7 = __float_as_uint(data.DebugData.y);
	p8 = __float_as_uint(data.DebugData.z);
	optixTraverse(params.Handle, RayOrigin, RayDirection, Tmin, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
		SBTOffset, SBTStride, MissSBTIndex,
		p0, p1, p2, p3, p4, p5, p6, p7, p8);
	optixReorder();
	optixInvoke(p0, p1, p2, p3, p4, p5, p6, p7, p8);
	data.Radience.x = __uint_as_float(p0);
	data.Radience.y = __uint_as_float(p1);
	data.Radience.z = __uint_as_float(p2);
	data.RecursionDepth = p3;
	data.Seed = p4;
	data.RayHitType = p5;
	data.DebugData.x = __uint_as_float(p6);
	data.DebugData.y = __uint_as_float(p7);
	data.DebugData.z = __uint_as_float(p8);
}
static __forceinline__ __device__ void optixTraceWithPerRayData(
	PerRayData& data,
	float3 RayOrigin,
	float3 RayDirection,
	uint SBTOffset,
	uint SBTStride,
	uint MissSBTIndex) {
	optixTraceWithPerRayData(
		data,
		RayOrigin,
		RayDirection,
		1e-4f,
		SBTOffset,
		SBTStride,
		MissSBTIndex);
}