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

extern "C" __global__ void __raygen__principled_bsdf(){
	// 没有显式递归的版本
	// 主体为一个循环，循环开始时根据上一个循环或rg产生的射线方向进行一次追踪。
	// 计算间接光的权重和直接光的辐射
	const uint3 idx = optixGetLaunchIndex();
	uint pixel_id = idx.y * RayTracingGlobalParams.Width + idx.x;
	const uint3 dim = optixGetLaunchDimensions();
	float3 RayOrigin, RayDirection;
	float2 jitter = Hammersley(RayTracingGlobalParams.FrameNumber % 32, 32);
	ComputeRayWithJitter(idx, dim, RayOrigin, RayDirection, jitter);

	float3 Weight=make_float3(1.0f);
	float3 Radiance=make_float3(0.0f);
	// 缓存每次追踪的直接辐照度，如果下一次间接反弹没有命中灯光，就直接叠加，如果下一次命中灯光，就进行MIS混合
	float2 MISWeightCache = make_float2(0);// x是Wf y是Wg
	float3 RadianceDirectCache = make_float3(0);
	uint RecursionDepth=0;
	// MIS需要知道发射的bxdf射线是否命中灯光
	// 但是当前的模式是计算并保存下一次追踪的方向，追踪的结果不在这一次递归中给出
	// 改为首先发射基础射线。在每帧追踪bxdf射线，返回bxdf结果后查看是否命中灯光，并将命中到的表面数据保存以便下一轮迭代使用

	// 首先发射primary ray
	HitInfo hitInfo;
	TraceRay(hitInfo,RayOrigin, RayDirection, TMIN, 0, 1, 0);
	if(hitInfo.surfaceType==Miss){
		RayTracingGlobalParams.IndirectOutputBuffer[pixel_id]=GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection);
		return;
	}
	else if(hitInfo.surfaceType==Light){
		// 第一次发射就命中灯光
		ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
		float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
		RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = LightColor;
		return;
	}
	float3 DebugColor=make_float3(0);
	for(;RecursionDepth<RayTracingGlobalParams.MaxRecursionDepth;RecursionDepth++){
		// 这次的着色点由上一次追踪给出
		
		// 加载命中点
		SurfaceData surfaceData;
		surfaceData.Clear();
		surfaceData.Load(hitInfo);
		
		// 假设只考虑漫射
		bool IsTransmission;
		float4 Noise4 = hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth));
		float4 Noise14 = hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth+0x11932287U));
		float3 BxdfWeight;
		float3 V = normalize(-RayDirection);
		RayDirection=PrincipledBsdf(RecursionDepth, surfaceData,make_float3(Noise4.x, Noise4.y, Noise4.z),V , BxdfWeight, IsTransmission);
		uint LightToSample = floor(RayTracingGlobalParams.LightListLength * Noise14.z);
		LightToSample = min(LightToSample, RayTracingGlobalParams.LightListLength - 1);

		if (surfaceData.Transmission < 1e-3f) {
			// 直接光
			float4 SampleResult = SampleLight(LightToSample, Noise14.x, Noise14.y, surfaceData.Position);
			float3 SamplePoint = make_float3(SampleResult.x, SampleResult.y, SampleResult.z);
			float3 RayDirDirectLight = normalize(SamplePoint - surfaceData.Position);
			float pdfSphereLight = SampleResult.w/ RayTracingGlobalParams.LightListLength;
			// 只有不是折射时进行NEE
			HitInfo hitInfoDirectLight;
			TraceRay(hitInfoDirectLight, surfaceData.Position, RayDirDirectLight, TMIN, 0, 1, 0);

			// shadow ray命中灯光
			float3 lightColor = hitInfoDirectLight.surfaceType == SurfaceType::Light ? GetColorFromAnyLight(LightToSample) : make_float3(0.0f);
			float3 BrdfDirect = EvalBsdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight));

			// 进行MIS f为Brdf采样， g为光源采样, X为Brdf样本，Y为光源样本
			float Pf_X = EvalPdf(surfaceData, V, RayDirection, false, normalize(V + RayDirection));
			float Pg_X = PdfLight(LightToSample, surfaceData.Position, RayDirection)/ RayTracingGlobalParams.LightListLength;
			// Wf
			MISWeightCache.x = Pf_X / (Pf_X + Pg_X);
			// Wg
			float Pg_Y = pdfSphereLight;
			float Pf_Y = EvalPdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight));
			MISWeightCache.y = Pg_Y / (Pf_Y + Pg_Y);
			RadianceDirectCache = Weight * lightColor * BrdfDirect / Pg_Y;
			
		}
		else {
			MISWeightCache.x = 1.0f;
			MISWeightCache.y = 0.0f;
			RadianceDirectCache = make_float3(0);
		}
		Weight *= BxdfWeight;
		// 立即追踪bxdf光线
		TraceRay(hitInfo, surfaceData.Position, RayDirection, TMIN, 0, 1, 0);

		// 判断是否miss
		if (hitInfo.surfaceType == Miss) {
			Radiance += Weight * GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection);
			Radiance += RadianceDirectCache;
			break;
		}
		else if (hitInfo.surfaceType == Light) {
			
			if (FetchLightIndex(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr)) == LightToSample) {
				// 递归中，除了阴影射线之外的射线命中灯光
				ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
				float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
				Radiance += RadianceDirectCache * MISWeightCache.y + MISWeightCache.x * Weight * LightColor;
				break;
			}
			else {
				ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
				float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
				Radiance += RadianceDirectCache+  Weight * LightColor;
				break;
			}
		}
		// 如果上次ShadowRay命中了灯光，而间接反弹没有命中，就直接叠加
		Radiance += RadianceDirectCache;
		
	}
	RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = Radiance;
}

extern "C" __global__ void __closesthit__fetch_hitinfo() {
	HitInfo hitInfo;
	hitInfo.PrimitiveID=optixGetPrimitiveIndex();
	hitInfo.SbtDataPtr=((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr;
	hitInfo.TriangleCentroidCoord=optixGetTriangleBarycentrics();
	hitInfo.surfaceType=((ModelData*)hitInfo.SbtDataPtr)->MaterialData->MaterialType==MaterialType::MATERIAL_OBJ ? Opaque:Light;
	SetPayLoad(hitInfo);
}

extern "C" __global__ void __miss__fetchMissInfo()
{
	HitInfo hitInfo;
	hitInfo.PrimitiveID=0xFFFFFFFF;
	hitInfo.SbtDataPtr=optixGetSbtDataPointer();
	hitInfo.TriangleCentroidCoord=make_float2(0.0f);
	hitInfo.surfaceType=Miss;
	SetPayLoad(hitInfo);
}


extern "C" __global__ void __intersection__sphere_light() {
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
extern "C" __global__ void __intersection__rectangle_light() {
	float3 ray_origin = optixGetWorldRayOrigin();
	float3 ray_direction = optixGetWorldRayDirection();
	float tmin = optixGetRayTmin();
	float tmax = optixGetRayTmax();
	ProceduralGeometryMaterialBuffer* data = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>();
	float3 p1, p2, p3, p4;
	float3 color;
	RectangleLight::DecodeRectangleLight(data->Elements,p1, p2, p3, p4, color);
	float t = RayIntersectWithRectangle(ray_origin, ray_direction, p1, p2, p3, p4, optixGetRayTmin(), optixGetRayTmax());
	if (!isnan(t)) {
		optixReportIntersection(t, 0);
	}
}
extern "C" __global__ void __closesthit__light(){
	HitInfo hitInfo;
	hitInfo.PrimitiveID= 0;
	hitInfo.SbtDataPtr=((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr;
	hitInfo.TriangleCentroidCoord=make_float2(0,0);
	hitInfo.surfaceType= SurfaceType::Light;
	SetPayLoad(hitInfo);
}