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
	float3 Radience=make_float3(0.0f);
	// ����ÿ��׷�ٵ�ֱ�ӷ��նȣ������һ�μ�ӷ���û�����еƹ⣬��ֱ�ӵ��ӣ������һ�����еƹ⣬�ͽ���MIS���
	float2 MISWeightCache = make_float2(0);// x��Wf y��Wg
	bool IsShadowRayHitLight = false;
	float3 RadianceDirectCache = make_float3(0);
	uint RecursionDepth=0;
	// MIS��Ҫ֪�������bxdf�����Ƿ����еƹ�
	// ���ǵ�ǰ��ģʽ�Ǽ��㲢������һ��׷�ٵķ���׷�ٵĽ��������һ�εݹ��и���
	// ��Ϊ���ȷ���������ߡ���ÿ֡׷��bxdf���ߣ�����bxdf�����鿴�Ƿ����еƹ⣬�������е��ı������ݱ����Ա���һ�ֵ���ʹ��

	// ���ȷ���primary ray
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
		// ��һ�η�������еƹ�
		ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
		float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
		RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = LightColor;
		return;
	}
	for(;RecursionDepth<RayTracingGlobalParams.MaxRecursionDepth;RecursionDepth++){
		// ��ε���ɫ������һ��׷�ٸ���
		// �ж��Ƿ�miss
		if(hitInfo.surfaceType==Miss){
			Radience += Weight * GetSkyBoxColor(hitInfo.SbtDataPtr, RayDirection);
			Radience += RadianceDirectCache;
			break;
		}
		else if(hitInfo.surfaceType==Light){
			// �ݹ��У�������Ӱ����֮����������еƹ�
			ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
			float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
			if (IsShadowRayHitLight) {
				// ����MIS���
				Radience+=RadianceDirectCache*MISWeightCache.y+ MISWeightCache.x*Weight * LightColor;
			}
			else {
				Radience += Weight * LightColor;
			}
			break;
		}
		// ����ϴ�ShadowRay�����˵ƹ⣬����ӷ���û�����У���ֱ�ӵ���
		if (IsShadowRayHitLight) {
			Radience += RadianceDirectCache;
		}
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

		if (surfaceData.Transmission < 1e-3f) {
			// ֱ�ӹ�
			float3 Color, LightCenter;
			float Radius;
			float* sphereLightData = FetchLightData(0U);
			DecodeSphereLight(sphereLightData, LightCenter, Radius, Color);
			float3 SamplePoint;
			float4 SampleResult = SampleSphereLight(Noise14.x, Noise14.y, surfaceData.Position, LightCenter, Radius, SamplePoint);
			float3 RayDirDirectLight = make_float3(SampleResult.x, SampleResult.y, SampleResult.z);
			float pdfSphereLight = SampleResult.w;
			// ֻ�в�������ʱ����NEE
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
				// shadow ray���еƹ�
				IsShadowRayHitLight = true;
				float3 lightColor = hitInfoDirectLight.surfaceType == SurfaceType::Light ? Color : make_float3(0.0f);
				float3 BrdfDirect = EvalBsdf(surfaceData, V, RayDirDirectLight, false, normalize(V + RayDirDirectLight));

				// ����MIS fΪBrdf������ gΪ��Դ����, XΪBrdf������YΪ��Դ����
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
		// ����׷��bxdf����
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