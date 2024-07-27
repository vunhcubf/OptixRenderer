#include "common.cuh"
#include "raytracing.cuh"
#include "bxdf.cuh"
#include "payload.cuh"
enum BxdfType {
	Dielectric,
	Metal,
	Glass
};
extern "C" __global__ void __closesthit__principled_bsdf()
{
	// ��ʿ���ԭ��bsdf
	// �����ֲ��ʹ��ɣ�����������ʺͽ���
	// �ֱ����������ߣ��������ߡ��������ߡ���������
	// һ��ֻ����һ�����ߣ�ʹ��mis���в���
	// ����͸��ͽ����Ⱦ������ֲ��ʵĻ�ϱ���
	// ������ߺͷ�����߹���brdf��������߹���btdf
	// �����ϰ����棬����ͷ�����ߵ�pdf��Ϊ0�������°����棬������ߵ�pdf��Ϊ0
	PerRayData Data = FetchPerRayDataFromPayLoad();
	if (GetModelDataPtr()->MaterialData->MaterialType == MATERIAL_AREALIGHT) {
		Data.Radience = RayTracingGlobalParams.areaLight.Color;
		Data.RayHitType = HIT_TYPE_LIGHT;
		SetPerRayDataForPayLoad(Data);
		return;
	}
	const uint3 Id = optixGetLaunchIndex();
	// ʹ�����̶ľ����Ƿ�ֹͣ
	//float roulette_ps = RayTracingGlobalParams.MaxRecursionDepth / (RayTracingGlobalParams.MaxRecursionDepth + 1.0f);
	//float rand_for_roulette = Rand(Data.Seed);
	//if (rand_for_roulette > roulette_ps) {
	//	Data.Radience = make_float3(0);
	//	Data.RayHitType = HIT_TYPE_SCENE;
	//	SetPerRayDataForPayLoad(Data);
	//	return;
	//}
	Data.RecursionDepth += 1;
	float3 radience_diff = make_float3(0);
	float3 radience_spec = make_float3(0);
	float3 BaseColor;
	float Roughness;
	float3 NormalMap;
	float Metallic;
	float Specular;
	float SpecularTint;
	{
		auto data = GetModelDataPtr();
		Specular = data->MaterialData->Specular;
		SpecularTint = data->MaterialData->SpecularTint;
	}
	float AO = 1.0f;
	ModelData* ModelDataptr = GetModelDataPtr();
	float2 uv = GetTexCoord();
	float3 N = GetNormal();
	float3 V = normalize(-optixGetWorldRayDirection());
	// ���㼸�η���
	float3 N_Geo;
	{
		float3 v1, v2, v3;
		GetTriangle(v1, v2, v3);
		N_Geo = normalize(cross(v1 - v2, v1 - v3));
	}
	if (ModelDataptr->MaterialData->BaseColorMap != NO_TEXTURE_HERE) {
		float4 tmp = SampleTexture2D<float4>(ModelDataptr->MaterialData->BaseColorMap, uv.x, uv.y);
		BaseColor = make_float3(tmp.x, tmp.y, tmp.z);
	}
	else {
		BaseColor = ModelDataptr->MaterialData->BaseColor;
	}
	BaseColor *= AO;
	if (ModelDataptr->MaterialData->ARMMap != NO_TEXTURE_HERE) {
		float4 tmp = SampleTexture2D<float4>(ModelDataptr->MaterialData->ARMMap, uv.x, uv.y);
		Roughness = tmp.y;
		Metallic = tmp.z;
		AO = tmp.x;
	}
	else {
		Roughness = ModelDataptr->MaterialData->Roughness;
		Metallic = ModelDataptr->MaterialData->Metallic;
	}
	Roughness = fmaxf(Roughness, 1e-3f);
	float Transmission = ModelDataptr->MaterialData->Transmission;
	float ior = ModelDataptr->MaterialData->Ior;
	ior = fmaxf(ior, 1.0001f);
	if (ModelDataptr->MaterialData->NormalMap != NO_TEXTURE_HERE) {
		float4 tmp = SampleTexture2D<float4>(ModelDataptr->MaterialData->NormalMap, uv.x, uv.y);
		NormalMap = make_float3(tmp.x, tmp.y, tmp.z);
		N = UseNormalMap(N, NormalMap, 1.0f);
	}
	float3 DebugColor = make_float3(0);
	float3 RayOrigin = GetPosition();

	// �����淨��ָ�����һ��
	bool in_surface = dot(N, V) >= 0.0f;
	float eta_i = in_surface ? 1.0f : ior;
	float eta_o = in_surface ? ior : 1.0f;
	float3 n_forward = in_surface ? N : -N;
	float3 h_forward;
	float3 n_geo_forward=dot(n_forward,N_Geo)>0.0f ? N_Geo : -N_Geo;
	float ps_re = 0.0f;
	float ps_tr = 0.0f;
	bool IsBxdfRayHitLight = false;

	
	float qs_glass = (1 - Metallic) * Transmission;
	float qs_dielectric = (1 - qs_glass);
	BxdfType MatToTrace;
	float3 L;
	// ����� | ���� | ����
	float rand_for_mat = Rand(Data.Seed);
	if (rand_for_mat < qs_dielectric) {
		MatToTrace = Dielectric;
	}
	else {
		MatToTrace = Glass;
	}
	float3 RadienceIndirect = make_float3(0);
	float rand_for_select_ray = Rand(Data.Seed);
	float p;
	if (Data.RecursionDepth < RayTracingGlobalParams.MaxRecursionDepth) {
		// �����ӹⲿ��
		switch (MatToTrace)
		{
		case Dielectric: {
			float qs_diffuse = 1 - Metallic;
			float qs_re = 1;
			float qs_sum = qs_diffuse + qs_re;
			qs_diffuse /= qs_sum;
			qs_re /= qs_sum;
			// ������о��������
			if (rand_for_select_ray < qs_diffuse) {
				// ����
				L = ImportanceSampleCosWeight(Data.Seed, N);
			}
			else {
				float3 H = ImportanceSampleGGX(Data.Seed, nullptr, GetModelDataPtr()->MaterialData->Roughness);
				float3 T, B;
				{
					GetTBNFromN(N, T, B);
					H = T * H.x + B * H.y + N * H.z;
					H = normalize(H);
				}
				h_forward = in_surface ? H : -H;
				L = normalize(2 * dot(h_forward, V) * h_forward - V);
			}
			// ����L��ָ�����
			//L = ClmapRayDir(n_geo_forward, L);
			float3 H = normalize(V + L);

			float pdf_di = saturate(dot(n_forward, L)) * REVERSE_PI;

			float pdf_m = DistributionGGX(abs(dot(N, H)), Roughness) * abs(dot(N, H));
			float jacob_reflection = 1.0f / (4 * abs(dot(H, L)));
			float pdf_re = pdf_m * jacob_reflection;

			PerRayData DataBxdf;
			DataBxdf.Radience = make_float3(0);
			DataBxdf.RecursionDepth = Data.RecursionDepth;
			DataBxdf.Seed = Data.Seed;

			optixTraceWithPerRayData(DataBxdf, RayOrigin+1e-3f* n_geo_forward, L, 0, 2, 0);
			IsBxdfRayHitLight = DataBxdf.RayHitType == HIT_TYPE_LIGHT;
			float HdotV = abs(dot(V, H));
			float3 Ctint = normalize(BaseColor);
			float3 Cs = lerp(0.08 * Specular * lerp(make_float3(1), Ctint, SpecularTint), BaseColor, Metallic);
			float3 Fs = Cs + (1 - Cs) * Pow5(1 - HdotV);
			//���߷ֲ�����
			float Ds = DistributionGGX(abs(dot(n_forward, H)), Roughness);
			//�ڱ���
			float Gs = Smith_G(n_forward, H, V, L, Roughness);
			float3 brdf_specular = Fs * Gs * Ds / abs(4 * dot(n_forward, V) * dot(n_forward, L));
			float3 brdf_diffuse = BaseColor * REVERSE_PI;
			float3 brdf = (1 - Metallic) * brdf_diffuse + brdf_specular;
			p = (pdf_re * qs_re + pdf_di * qs_diffuse);
			RadienceIndirect = DataBxdf.Radience * saturate(dot(n_forward, L)) * brdf / p;
			if (Data.RecursionDepth == 1) {
				Data.DebugData = DataBxdf.Radience;
			}
		}
					   break;
		case Glass: {
			float3 H = ImportanceSampleGGX(Data.Seed, nullptr, GetModelDataPtr()->MaterialData->Roughness);
			float3 T, B;
			{
				GetTBNFromN(N, T, B);
				H = T * H.x + B * H.y + N * H.z;
				H = normalize(H);
			}
			h_forward = in_surface ? H : -H;

			bool internal_reflection = false;
			// ��������������ο����������㷢��ʲô����
			// �����������Ϊ�����֣������͸�䣬brdf��btdf
			// ʹ��MISͬʱ���������ֲ�����ʹ��ӳ���ķ�������Ϊÿ�ֲ����ĸ��ʣ�
			// ���������qsΪӳ���ķ�������psΪpm*p_re
			// ���������qsΪ1-ӳ���ķ�����, psΪpm*p_trans
			// ��brdf�в�ʡ��NdotL��
			// ������֮ǰ���������������⣬��iorΪ1ʱ��eta_i * dot(V, h_forward) + eta_o * dot(L, h_forward)�ϸ�Ϊ0��Ӧ��ior��Сֵ����Ϊ1.0001f
			// ��С���߳�������Ϊ1e-4������Ϊ1e-3fʱ�ڲ����Ľ�����������
			float fs;
			float c = abs(dot(V, h_forward));
			float g = pow2(eta_o / eta_i) - 1.0f + c * c;
			if (g < 0.0f) {
				// ȫ����
				fs = 1.0f;
			}
			else {
				g = sqrt(g);
				fs = 0.5f * pow2((g - c) / (g + c)) * (1 + pow2((c * (g + c) - 1) / (c * (g - c) + 1)));
			}
			fs = saturate(fs);
			float qs_re = lerp(0.1, 0.9, fs);
			float qs_tr = 1 - qs_re;

			float rand_num = rand_for_select_ray;
			bool is_reflect;
			if (rand_num < qs_re) {
				// ����
				is_reflect = true;
				L = normalize(2 * dot(h_forward, V) * h_forward - V);
			}
			else {
				//����
				is_reflect = false;
				L = refract(-V, h_forward, eta_i / eta_o, nullptr);
			}

			float pdf_m = DistributionGGX(abs(dot(h_forward, n_forward)), Roughness) * abs(dot(h_forward, n_forward));
			float jacob_re = 1.0f / (4 * abs(dot(H, L)));
			ps_re = pdf_m * jacob_re;
			float jacob_tr = eta_o * eta_o * abs(dot(L, h_forward)) / pow2(eta_i * dot(V, h_forward) + eta_o * dot(L, h_forward));
			ps_tr = pdf_m * jacob_tr;

			PerRayData DataBxdf;
			DataBxdf.Radience = make_float3(0);
			DataBxdf.RecursionDepth = Data.RecursionDepth;
			DataBxdf.Seed = Data.Seed;
			optixTraceWithPerRayData(DataBxdf, RayOrigin, L, 1e-4f, 0, 2, 0);
			////���߷ֲ�����
			float Ds = DistributionGGX(h_forward, n_forward, Roughness);
			//�ڱ���
			float Gs = Smith_G(n_forward, h_forward, V, L, Roughness);
			float3 brdf = BaseColor * fs * Gs * Ds / abs(4 * dot(n_forward, V) * dot(n_forward, L));

			float3 numerator = sqrt(BaseColor) * (1 - fs) * Ds * Gs * abs(dot(h_forward, L) * dot(h_forward, V)) * eta_o * eta_o;
			float denominator = abs(dot(n_forward, V) * dot(n_forward, L)) * pow2(eta_i * dot(V, h_forward) + eta_o * dot(L, h_forward));
			float3 btdf = numerator / denominator;
			if (is_reflect) {
				RadienceIndirect = DataBxdf.Radience * saturate(dot(n_forward, L)) * brdf / (ps_re * qs_re);
			}
			else {
				RadienceIndirect = DataBxdf.Radience * saturate(-dot(n_forward, L)) * btdf / (ps_tr * qs_tr);
			}
			
		}
			break;
		default:
			break;
		}
		
	}
	
	// ֱ�ӹ�
	float3 RadienceDirect = make_float3(0);
	float pdf_light = 1 / RayTracingGlobalParams.areaLight.Area;
	if (MatToTrace != Glass) {
		float3 SamplePoint = RandomSamplePointOnLight(Data.Seed);
		float3 ray_dir_direct = normalize(SamplePoint - RayOrigin);

		
		if (dot(GetNormal(), ray_dir_direct)>0.0f && ray_dir_direct.z > 1e-2f) {
			PerRayData DataDirect;
			optixTraceWithPerRayData(DataDirect, RayOrigin, ray_dir_direct,1e-4f, 1, 2, 0);

			float3 H_d = normalize(V + ray_dir_direct);
			float HdotV = abs(dot(V, H_d));
			float3 Ctint = normalize(BaseColor);
			float3 Cs = lerp(0.08 * Specular * lerp(make_float3(1), Ctint, SpecularTint), BaseColor, 1);
			float3 Fs = Cs + (1 - Cs) * Pow5(1 - HdotV);
			//���߷ֲ�����
			float Ds = DistributionGGX(n_forward, H_d, Roughness);
			//�ڱ���
			float Gs = Smith_G(n_forward, H_d, V, ray_dir_direct, Roughness);
			float3 brdf = Fs * Gs * Ds / abs(4 * dot(N, V) * dot(N, ray_dir_direct)) + (1 - Metallic) * BaseColor * REVERSE_PI;

			float Dw = saturate(ray_dir_direct.z) / squared_length(RayOrigin - SamplePoint);
			RadienceDirect = DataDirect.Radience * Dw * brdf * saturate(dot(N, ray_dir_direct)) / pdf_light;
		}
		if (IsBxdfRayHitLight) {
			Data.Radience = (RadienceDirect * pdf_light + RadienceIndirect * p) / (p + pdf_light);
		}
		else {
			Data.Radience = RadienceDirect + RadienceIndirect;
		}
	}
	else {
		Data.Radience = RadienceIndirect;
	}
	
	Data.Radience = fmaxf(Data.Radience,make_float3(1e-3f));
	Data.RayHitType = HIT_TYPE_SCENE;
	SetPerRayDataForPayLoad(Data);
}

extern "C" __global__ void __closesthit__occluded() {
	SbtDataStruct* HitGroupDataPtr = (SbtDataStruct*)optixGetSbtDataPointer();
	ModelData* modeldata_ptr = (ModelData*)HitGroupDataPtr->DataPtr;
	PerRayData Data = FetchPerRayDataFromPayLoad();
	if (modeldata_ptr->MaterialData->MaterialType == MATERIAL_AREALIGHT) {
		Data.Radience = RayTracingGlobalParams.areaLight.Color;
		Data.RayHitType = HIT_TYPE_LIGHT;
	}
	else {
		Data.Radience = make_float3(0);
		Data.RayHitType = HIT_TYPE_SCENE;
	}
	Data.DebugData = make_float3(0.5);
	SetPerRayDataForPayLoad(Data);
}

extern "C" __global__ void __raygen__principled_bsdf(){
	// û����ʽ�ݹ�İ汾
	// ����Ϊһ��ѭ����ѭ����ʼʱ������һ��ѭ����rg���������߷������һ��׷�١�
	// �����ӹ��Ȩ�غ�ֱ�ӹ�ķ���
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	float3 RayOrigin, RayDirection;
	float2 jitter = Hammersley(RayTracingGlobalParams.FrameNumber % 32, 32);
	ComputeRayWithJitter(idx, dim, RayOrigin, RayDirection, jitter);

	float3 Weight=make_float3(1.0f);
	float3 Radience=make_float3(0.0f);
	uint RecursionDepth=0;

	float3 DebugColor;
	bool FutureBxdfRayHitLight=false;
	float LastMISWeight=1.0f;
	// MIS��Ҫ֪�������bxdf�����Ƿ����еƹ�
	// ���ǵ�ǰ��ģʽ�Ǽ��㲢������һ��׷�ٵķ���׷�ٵĽ��������һ�εݹ��и���
	// ��Ϊ���ȷ���������ߡ���ÿ֡׷��bxdf���ߣ�����bxdf�����鿴�Ƿ����еƹ⣬�������е��ı������ݱ����Ա���һ�ֵ���ʹ��
	for(;RecursionDepth<RayTracingGlobalParams.MaxRecursionDepth;RecursionDepth++){
		// ��׷�ٹ���
		HitInfo hitInfo;
		TraceRay(hitInfo, RayOrigin, RayDirection,1e-3f, 0, 2, 0);
		if(hitInfo.PrimitiveID==0xFFFFFFFF){
			// δ����
			Radience+=Weight*make_float3(0.5f);
			break;
		}
		else if(((ModelData*)hitInfo.SbtDataPtr)->MaterialData->MaterialType==MATERIAL_AREALIGHT){
			Radience+=Weight*RayTracingGlobalParams.areaLight.Color;
			break;
		}
		// �������е�
		SurfaceData surfaceData;
		surfaceData.Load(hitInfo);
		
		// ����ֻ��������
		float4 bluenoise;
		SAMPLE_BLUENOISE_4D(bluenoise);
		uint seed=RayTracingGlobalParams.Seed+idx.x*+idx.y*RayTracingGlobalParams.Width+(RayTracingGlobalParams.FrameNumber)%0xFFFFFFFF;
		bluenoise.x=Rand(seed);
		bluenoise.y=Rand(seed);
		bluenoise.z=Rand(seed);
		RayDirection=ImportanceSampleCosWeight(make_float2(bluenoise.x,bluenoise.y),surfaceData.Normal);
		RayOrigin=surfaceData.Position;

		// ֱ�ӹ�
		float3 SamplePoint = RandomSamplePointOnLight(make_float2(bluenoise.y,bluenoise.z));
		float3 RayDirDirectLight = normalize(SamplePoint - RayOrigin);
		float3 RadienceDirect=make_float3(0.0f);

		
		if (dot(surfaceData.Normal, RayDirDirectLight) > 1e-2f && RayDirDirectLight.z > 1e-2f) {
			HitInfo hitInfoDirectLight;
			TraceRay(hitInfoDirectLight, RayOrigin, RayDirDirectLight,1e-3f, 0, 2, 0);
			HitLight=((ModelData*)(hitInfoDirectLight.SbtDataPtr))->MaterialData->MaterialType==MATERIAL_AREALIGHT;
			float3 lightColor=HitLight ? RayTracingGlobalParams.areaLight.Color : make_float3(0.0f);
			float Dw = RayTracingGlobalParams.areaLight.Area * saturate(dot(surfaceData.Normal, RayDirDirectLight) + 1e-4f) * saturate(RayDirDirectLight.z) / squared_length(RayOrigin - SamplePoint);
			RadienceDirect = lightColor * Dw * surfaceData.BaseColor * REVERSE_PI;
		}

		// MIS
		float PdfDiffuse=abs(dot(surfaceData.Normal,RayDirection))*REVERSE_PI;
		float PdfLight = 1 / RayTracingGlobalParams.areaLight.Area;

		float MISWeight=1.0f;
		if (HitLight) {
			MISWeight=PdfDiffuse / (PdfDiffuse + PdfLight);
			RadienceDirect=(RadienceDirect * PdfLight) / (PdfDiffuse + PdfLight);
		}
		if(RecursionDepth==0){
			DebugColor=make_float3(MISWeight);
		}
		Radience+=Weight*RadienceDirect;
		Weight*=surfaceData.BaseColor*MISWeight;
	}
	uint pixel_id = idx.y * RayTracingGlobalParams.Width + idx.x;
	RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = DebugColor;
}

extern "C" __global__ void __closesthit__fetch_hitinfo() {
	HitInfo hitInfo;
	hitInfo.PrimitiveID=optixGetPrimitiveIndex();
	hitInfo.SbtDataPtr=((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr;
	hitInfo.TriangleCentroidCoord=optixGetTriangleBarycentrics();
	SetPayLoad(hitInfo);
}

extern "C" __global__ void __miss__fetchMissInfo()
{
	HitInfo hitInfo;
	hitInfo.PrimitiveID=0xFFFFFFFF;
	hitInfo.SbtDataPtr=((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr;
	hitInfo.TriangleCentroidCoord=make_float2(0.0f);
	SetPayLoad(hitInfo);
}