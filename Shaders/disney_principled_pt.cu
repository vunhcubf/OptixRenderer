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
	float3 Radience=make_float3(0.0f);
	// 缓存每次追踪的直接辐照度，如果下一次间接反弹没有命中灯光，就直接叠加，如果下一次命中灯光，就进行MIS混合
	float2 MISWeightCache = make_float2(0);// x是Wf y是Wg
	bool IsShadowRayHitLight = false;
	float3 RadianceDirectCache = make_float3(0);
	uint RecursionDepth=0;
	// MIS需要知道发射的bxdf射线是否命中灯光
	// 但是当前的模式是计算并保存下一次追踪的方向，追踪的结果不在这一次递归中给出
	// 改为首先发射基础射线。在每帧追踪bxdf射线，返回bxdf结果后查看是否命中灯光，并将命中到的表面数据保存以便下一轮迭代使用

	// 首先发射primary ray
	HitInfo hitInfo;
	TraceRay(hitInfo,RayOrigin, RayDirection,1e-3f, 0, 1, 0);
	if(hitInfo.surfaceType==Miss){
		MissData* data = (MissData*)hitInfo.SbtDataPtr;
		float2 SkyBoxUv = GetSkyBoxUv(RayDirection);
		if (IsTextureViewValid(data->SkyBox)) {
			float4 skybox = SampleTexture2DRuntimeSpecific(data->SkyBox, SkyBoxUv.x, SkyBoxUv.y);
			RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = make_float3(skybox.x, skybox.y, skybox.z);
		}
		else {
			RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = data->BackgroundColor * data->SkyBoxIntensity;
		}
		return;
	}
	else if(hitInfo.surfaceType==Light){
		// 第一次发射就命中灯光
		ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
		float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
		RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = LightColor;
		return;
	}
	for(;RecursionDepth<RayTracingGlobalParams.MaxRecursionDepth;RecursionDepth++){
		// 这次的着色点由上一次追踪给出
		// 判断是否miss
		if(hitInfo.surfaceType==Miss){
			Radience += Weight * GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection);
			Radience += RadianceDirectCache;
			break;
		}
		else if(hitInfo.surfaceType==Light){
			// 递归中，除了阴影射线之外的射线命中灯光
			ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
			float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
			if (IsShadowRayHitLight) {
				// 进行MIS混合
				Radience+=RadianceDirectCache*MISWeightCache.y+ MISWeightCache.x*Weight * LightColor;
			}
			else {
				Radience += Weight * LightColor;
			}
			break;
		}
		// 如果上次ShadowRay命中了灯光，而间接反弹没有命中，就直接叠加
		if (IsShadowRayHitLight) {
			Radience += RadianceDirectCache;
		}
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

		if (surfaceData.Transmission < 1e-3f) {
			// 直接光
			float3 Color, LightCenter;
			float Radius;
			float* sphereLightData = FetchLightData(0U);
			DecodeSphereLight(sphereLightData, LightCenter, Radius, Color);
			float3 SamplePoint;
			float4 SampleResult = SampleSphereLight(Noise14.x, Noise14.y, surfaceData.Position, LightCenter, Radius, SamplePoint);
			float3 RayDirDirectLight = make_float3(SampleResult.x, SampleResult.y, SampleResult.z);
			float pdfSphereLight = SampleResult.w;
			// 只有不是折射时进行NEE
			HitInfo hitInfoDirectLight;
			TraceRay(hitInfoDirectLight, surfaceData.Position, RayDirDirectLight, 1e-3f, 0, 1, 0);
			if (hitInfoDirectLight.surfaceType != SurfaceType::Light) {
				Weight *= BxdfWeight;
				IsShadowRayHitLight = false;
				MISWeightCache.x = 1.0f;
				MISWeightCache.y = 0.0f;
				RadianceDirectCache = make_float3(0);
			}
			else {
				// shadow ray命中灯光
				IsShadowRayHitLight = true;
				float3 lightColor = hitInfoDirectLight.surfaceType == SurfaceType::Light ? Color : make_float3(0.0f);
				float3 BrdfDirect = EvalBsdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight));

				// 进行MIS f为Brdf采样， g为光源采样, X为Brdf样本，Y为光源样本
				float Pf_X = EvalPdf(surfaceData, V, RayDirection, false, normalize(V + RayDirection));
				float Pg_X = PdfSphereLight(surfaceData.Position,LightCenter,Radius,RayDirection);
				// Wf
				MISWeightCache.x = Pf_X / (Pf_X + Pg_X);
				// Wg
				float Pg_Y = pdfSphereLight;
				float Pf_Y = EvalPdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight));
				MISWeightCache.y = Pg_Y / (Pf_Y + Pg_Y);
				RadianceDirectCache = Weight * lightColor * BrdfDirect / Pg_Y;
				if (optixGetLaunchIndex().x == 600 && optixGetLaunchIndex().y == 600) {
					printf("luminance %f,\t%f,\t%f\n", RadianceDirectCache.x, RadianceDirectCache.y, RadianceDirectCache.z);
					printf("brdf %f,\t%f,\t%f\n", BrdfDirect.x, BrdfDirect.y, BrdfDirect.z);
					printf("pdf %f\n", Pg_Y);
					printf("weight %f,\t%f,\t%f\n", Weight.x, Weight.y, Weight.z);
					printf("\n");
				}
				Weight *= BxdfWeight;
			}
		}
		else {
			Weight *= BxdfWeight;
			MISWeightCache.x = 1.0f;
			MISWeightCache.y = 0.0f;
			IsShadowRayHitLight = false;
			RadianceDirectCache = make_float3(0);
		}
		// 立即追踪bxdf光线
		TraceRay(hitInfo, surfaceData.Position, RayDirection, 1e-3f, 0, 1, 0);
	}
	RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] =  Radience;
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
    float3 sphere_center = pos;
    float sphere_radius = radius;
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
        if (t > tmin && t < tmax) {
            optixReportIntersection(t, 0); 
        }
    }
}
extern "C" __global__ void __closesthit__sphere_light(){
	HitInfo hitInfo;
	hitInfo.PrimitiveID= 0;
	hitInfo.SbtDataPtr=((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr;
	hitInfo.TriangleCentroidCoord=make_float2(0,0);
	hitInfo.surfaceType= SurfaceType::Light;
	SetPayLoad(hitInfo);
}