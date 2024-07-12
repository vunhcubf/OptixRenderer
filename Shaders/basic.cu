#include "common.cuh"
#include "raytracing.cuh"
#include "bxdf.cuh"
extern "C" __global__ void __exception__shader() {
	const unsigned int code = optixGetExceptionCode();
	printf("Exception code: %u\n", code);
}
extern "C" __global__ void __raygen__primary_ray()
{
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	float3 RayOrigin, RayDirection, Radience;
	float2 jitter = Hammersley(params.FrameNumber % 32, 32);
	ComputeRayWithJitter(idx, dim, RayOrigin, RayDirection, jitter);
	Radience = make_float3(0);
	uint spp = params.Spp;
	for (int i_spp = 0; i_spp < spp; i_spp++) {
		PerRayData Data;
		Data.DebugData = make_float3(0.0f);
		Data.Radience = make_float3(0);
		Data.RayHitType = HIT_TYPE_SCENE;
		Data.RecursionDepth = 0;
		Data.Seed = tea<5>(idx.y * params.Width + idx.x, params.Seed + 1919810 * i_spp);
		optixTraceWithPerRayDataReordered(Data, RayOrigin, RayDirection,1e-3f, 0, 2, 0);
		Radience += Data.Radience;
	}
	uint pixel_id = idx.y * params.Width + idx.x;
	params.IndirectOutputBuffer[pixel_id] = Radience / spp;
}

extern "C" __global__ void __miss__sky()
{
	PerRayData Data = FetchPerRayDataFromPayLoad();

	MissData* data = (MissData*)optixGetSbtDataPointer();
	float3 RayDir = optixGetWorldRayDirection();
	float2 SkyBoxUv = GetSkyBoxUv(RayDir);
	if (data->SkyBox != NO_TEXTURE_HERE) {
		float4 skybox = SampleTexture2D<float4>(data->SkyBox, SkyBoxUv.x, SkyBoxUv.y);
		Data.Radience=make_float3(skybox.x, skybox.y, skybox.z);
		//Data.Radience = fminf(make_float3(skybox.x, skybox.y, skybox.z), make_float3(20.0f)) * data->SkyBoxIntensity;
	}
	else {
		Data.Radience = data->BackgroundColor * data->SkyBoxIntensity;
	}
	if (Data.RecursionDepth == 0) {
		Data.DebugData = Data.Radience;
	}
	else {
		Data.DebugData = make_float3(0, 0, 0.5);
	}
	SetPerRayDataForPayLoad(Data);
}