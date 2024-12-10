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
			MissData* data = (MissData*)hitInfo.SbtDataPtr;
			float2 SkyBoxUv = GetSkyBoxUv(RayDirection);
			if (IsTextureViewValid(data->SkyBox)) {
				float4 skybox = SampleTexture2DRuntimeSpecific(data->SkyBox, SkyBoxUv.x, SkyBoxUv.y);
				Radience+=Weight*make_float3(skybox.x, skybox.y, skybox.z);
			}
			else {
				Radience+=Weight*data->BackgroundColor * data->SkyBoxIntensity;
			}
			break;
		}
		else if(hitInfo.surfaceType==Light){
			// �ݹ��У�������Ӱ����֮����������еƹ�
			ProceduralGeometryMaterialBuffer* proceduralMatPtr = GetSbtDataPointer<ProceduralGeometryMaterialBuffer>(hitInfo.SbtDataPtr);
			float3 LightColor = GetColorFromAnyLight(FetchLightData(proceduralMatPtr));
			Radience+=Weight* LightColor;
			break;
		}
		// �������е�
		SurfaceData surfaceData;
		surfaceData.Clear();
		surfaceData.Load(hitInfo);
		
		// ����ֻ��������
		bool TraceGlass;
		float4 Noise4 = hash44(make_uint4(idx.x, idx.y, RayTracingGlobalParams.FrameNumber, RecursionDepth));
		float3 BxdfWeight;
		PrincipledBsdf(RecursionDepth, surfaceData, RayDirection, BxdfWeight, TraceGlass);
		Weight *= BxdfWeight;
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
	hitInfo.PrimitiveID= 0xFFFFFF0F;
	hitInfo.SbtDataPtr=((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr;
	hitInfo.TriangleCentroidCoord=make_float2(0,0);
	hitInfo.surfaceType= SurfaceType::Light;
	SetPayLoad(hitInfo);
}