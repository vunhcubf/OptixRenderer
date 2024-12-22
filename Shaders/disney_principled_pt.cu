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
	// û����ʽ�ݹ�İ汾
	// ����Ϊһ��ѭ����ѭ����ʼʱ������һ��ѭ����rg���������߷������һ��׷�١�
	// �����ӹ��Ȩ�غ�ֱ�ӹ�ķ���
	const uint3 idx = optixGetLaunchIndex();
	uint pixel_id = idx.y * RayTracingGlobalParams.Width + idx.x;
	const uint3 dim = optixGetLaunchDimensions();
	float3 RayOrigin, RayDirection;
	float2 jitter = Hammersley(RayTracingGlobalParams.FrameNumber % 32, 32);
	ComputeRayWithJitter(idx, dim, RayOrigin, RayDirection, jitter);

	float3 Weight=make_float3(1.0f);
	float3 Radiance=make_float3(0.0f);
	// ����ÿ��׷�ٵ�ֱ�ӷ��նȣ������һ�μ�ӷ���û�����еƹ⣬��ֱ�ӵ��ӣ������һ�����еƹ⣬�ͽ���MIS���
	float2 MISWeightCache = make_float2(0);// x��Wf y��Wg
	float3 RadianceDirectCache = make_float3(0);
	uint RecursionDepth=0;
	// MIS��Ҫ֪�������bxdf�����Ƿ����еƹ�
	// ���ǵ�ǰ��ģʽ�Ǽ��㲢������һ��׷�ٵķ���׷�ٵĽ��������һ�εݹ��и���
	// ��Ϊ���ȷ���������ߡ���ÿ֡׷��bxdf���ߣ�����bxdf�����鿴�Ƿ����еƹ⣬�������е��ı������ݱ����Ա���һ�ֵ���ʹ��

	// ���ȷ���primary ray
	HitInfo hitInfo;
	TraceRay(hitInfo,RayOrigin, RayDirection, TMIN, 0, 1, 0);
	if(hitInfo.surfaceType==Miss){
		RayTracingGlobalParams.IndirectOutputBuffer[pixel_id]=GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection);
		return;
	}
	else if(hitInfo.surfaceType==Light){
		// ��һ�η�������еƹ�
		ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
		float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
		RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = LightColor;
		return;
	}
	float3 DebugColor=make_float3(0);
	for(;RecursionDepth<RayTracingGlobalParams.MaxRecursionDepth;RecursionDepth++){
		// ��ε���ɫ������һ��׷�ٸ���
		
		// �������е�
		SurfaceData surfaceData;
		surfaceData.Clear();
		surfaceData.Load(hitInfo);
		
		// ����ֻ��������
		bool IsTransmission;
		float4 Noise4 = hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth));
		float4 Noise14 = hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth+0x11932287U));
		float3 BxdfWeight;
		float3 V = normalize(-RayDirection);
		RayDirection=PrincipledBsdf(RecursionDepth, surfaceData,make_float3(Noise4.x, Noise4.y, Noise4.z),V , BxdfWeight, IsTransmission);
		uint LightToSample = floor(RayTracingGlobalParams.LightListLength * Noise14.z);
		LightToSample = min(LightToSample, RayTracingGlobalParams.LightListLength - 1);

		if (surfaceData.Transmission < 1e-3f) {
			// ֱ�ӹ�
			float4 SampleResult = SampleLight(LightToSample, Noise14.x, Noise14.y, surfaceData.Position);
			float3 SamplePoint = make_float3(SampleResult.x, SampleResult.y, SampleResult.z);
			float3 RayDirDirectLight = normalize(SamplePoint - surfaceData.Position);
			float pdfSphereLight = SampleResult.w/ RayTracingGlobalParams.LightListLength;
			// ֻ�в�������ʱ����NEE
			HitInfo hitInfoDirectLight;
			TraceRay(hitInfoDirectLight, surfaceData.Position, RayDirDirectLight, TMIN, 0, 1, 0);

			// shadow ray���еƹ�
			float3 lightColor = hitInfoDirectLight.surfaceType == SurfaceType::Light ? GetColorFromAnyLight(LightToSample) : make_float3(0.0f);
			float3 BrdfDirect = EvalBsdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight));

			// ����MIS fΪBrdf������ gΪ��Դ����, XΪBrdf������YΪ��Դ����
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
		// ����׷��bxdf����
		TraceRay(hitInfo, surfaceData.Position, RayDirection, TMIN, 0, 1, 0);

		// �ж��Ƿ�miss
		if (hitInfo.surfaceType == Miss) {
			Radiance += Weight * GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection);
			Radiance += RadianceDirectCache;
			break;
		}
		else if (hitInfo.surfaceType == Light) {
			
			if (FetchLightIndex(GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr)) == LightToSample) {
				// �ݹ��У�������Ӱ����֮����������еƹ�
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
		// ����ϴ�ShadowRay�����˵ƹ⣬����ӷ���û�����У���ֱ�ӵ���
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