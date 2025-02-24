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
	float2 jitter = Hammersley(RayTracingGlobalParams.FrameNumber % 32, 32);
	ComputeRayWithJitter(idx, dim, RayOrigin, RayDirection, jitter);

	float3 Weight = make_float3(1.0f);
	float3 Radiance = make_float3(0.0f);
	// 缓存每次追踪的直接辐照度，如果下一次间接反弹没有命中灯光，就直接叠加，如果下一次命中灯光，就进行MIS混合
	uint RecursionDepth = 0;
	// MIS需要知道发射的bxdf射线是否命中灯光
	// 但是当前的模式是计算并保存下一次追踪的方向，追踪的结果不在这一次递归中给出
	// 改为首先发射基础射线。在每帧追踪bxdf射线，返回bxdf结果后查看是否命中灯光，并将命中到的表面数据保存以便下一轮迭代使用

	// 首先发射primary ray
	HitInfo hitInfo;
	TraceRay(hitInfo, RayOrigin, RayDirection, TMIN, 0, 1, 0);
	if (hitInfo.surfaceType == Miss) {
		RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection);
		return;
	}
	else if (hitInfo.surfaceType == Light) {
		// 第一次发射就命中灯光
		ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
		float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
		RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = LightColor;
		return;
	}
	float3 DebugColor = make_float3(0);
	for (; RecursionDepth < RayTracingGlobalParams.MaxRecursionDepth; RecursionDepth++) {
		// 这次的着色点由上一次追踪给出

		// 加载命中点
		SurfaceData surfaceData;
		surfaceData.Clear();
		surfaceData.Load(hitInfo);

		// 假设只考虑漫射
		bool IsTransmission;
		float Pf_X;
		float3 BsdfIndirect;
		float4 Noise4 = hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth));
		float4 Noise14 = hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth + 0x11932287U));
		float3 V = normalize(-RayDirection);
		{
			float3 HForward;
			RayDirection = SampleBsdf(surfaceData, make_float3(Noise4.x, Noise4.y, Noise4.z), V, IsTransmission, HForward);
			Pf_X = EvalPdf(surfaceData, V, RayDirection, IsTransmission, HForward);
			BsdfIndirect = EvalBsdf(surfaceData, V, RayDirection, IsTransmission, HForward);
		}
		// Pf_X为bsdf样本的bsdf概率
		// Pf_Y为灯光样本的bsdf概率
		// Pg_X为bsdf样本的灯光概率
		// Pg_Y为灯光样本的灯光概率
		// 直接光照要追踪每一盏灯光
		// 针对每个样本的收集公式为 Bsdf*Li/(sum{pdf})
		TraceRay(hitInfo, surfaceData.Position, RayDirection, TMIN, 0, 1, 0);
		// 只有非透明才收集直接光照

		float3 IrradianceDirect = make_float3(0);
		float3 IrradianceIndirect = make_float3(0);
		float3 WeightNew = make_float3(0);
		bool terminateRay = false;
		if (!IsTransmission && RayTracingGlobalParams.consoleOptions->debugMode == ConsoleDebugMode::MIS) {
			// 直接光
			// 随机选择光源进行评估，分层抽样
			// 若选择了一个灯光，就忽略其他灯光
			uint LightToSample = (uint)floor(frac(Noise14.z) * RayTracingGlobalParams.LightListLength);
			LightToSample = min(RayTracingGlobalParams.LightListLength - 1, LightToSample);
			float3 LiDirect, BrdfDirect;

			float4 SampleResult = SampleLight(LightToSample, Noise14.x, Noise14.y, surfaceData.Position);
			float3 SamplePoint = make_float3(SampleResult.x, SampleResult.y, SampleResult.z);
			float3 RayDirDirectLight = normalize(SamplePoint - surfaceData.Position);
			HitInfo hitInfoDirectLight;
			TraceRay(hitInfoDirectLight, surfaceData.Position, RayDirDirectLight, TMIN, 0, 1, 0);
			LiDirect = hitInfoDirectLight.surfaceType == SurfaceType::Light ? GetColorFromAnyLight(FetchLightData(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfoDirectLight.SbtDataPtr))) : make_float3(0.0f);
			BrdfDirect = EvalBsdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight));

			// 进行MIS f为Brdf采样， g为光源采样, X为Brdf样本，Y为光源样本
			// 遍历所有灯光计算pdf

			float Pf_Y = EvalPdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight));
			float WeightSum = Pf_Y;
			for (uint light = 0; light < RayTracingGlobalParams.LightListLength; light++) {
				float Pg_Y = PdfLight(light, surfaceData.Position, RayDirDirectLight) / RayTracingGlobalParams.LightListLength;
				WeightSum += Pg_Y;
			}
			IrradianceDirect += BrdfDirect * LiDirect / WeightSum;

			// 收集间接光照
			WeightSum = Pf_X;
			for (uint light = 0; light < RayTracingGlobalParams.LightListLength; light++) {
				float Pg_X = PdfLight(light, surfaceData.Position, RayDirection);
				WeightSum += Pg_X;
			}
			if (hitInfo.surfaceType == SurfaceType::Miss) {
				IrradianceIndirect = ASSERT_VALID(GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection));
				IrradianceIndirect *= ASSERT_VALID(BsdfIndirect / WeightSum);
				terminateRay = true;
			}
			// 间接辐照度收集灯光光照
			else if (hitInfo.surfaceType == SurfaceType::Light) {
				float3 LightColor = GetColorFromAnyLight(FetchLightData(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr)));
				IrradianceIndirect = LightColor * BsdfIndirect / WeightSum;
				terminateRay = true;
			}
			// 命中物体，继续渲染，不收集光照
			else {
				WeightNew = Weight * BsdfIndirect / WeightSum;
			}
			Radiance += ASSERT_VALID(Weight * IrradianceDirect);
			if (terminateRay) {
				Radiance += ASSERT_VALID(Weight * IrradianceIndirect);
			}
		}
		else {
			//没有直接光
			if (hitInfo.surfaceType == SurfaceType::Miss) {
				IrradianceIndirect = GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection);
				IrradianceIndirect *= BsdfIndirect / Pf_X;
				terminateRay = true;
			}
			// 间接辐照度收集灯光光照
			else if (hitInfo.surfaceType == SurfaceType::Light) {
				IrradianceIndirect = GetColorFromAnyLight(FetchLightData(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr)));
				IrradianceIndirect *= BsdfIndirect / Pf_X;
				terminateRay = true;
			}
			else {
				WeightNew = Weight * BsdfIndirect / Pf_X;
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