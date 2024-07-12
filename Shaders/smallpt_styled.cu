#include "common.cuh"
#include "raytracing.cuh"
#include "bxdf.cuh"

// 漫射
extern "C" __global__ void __closesthit__diffuse()
{

	PerRayData Data = FetchPerRayDataFromPayLoad();
	if (GetModelDataPtr()->MaterialData->MaterialType == MATERIAL_AREALIGHT) {
		Data.Radience = params.areaLight.Color;
		Data.RayHitType = HIT_TYPE_LIGHT;
		SetPerRayDataForPayLoad(Data);
		return;
	}
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
	float3 N_Geo = N;
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
	if (ModelDataptr->MaterialData->NormalMap != NO_TEXTURE_HERE) {
		float4 tmp = SampleTexture2D<float4>(ModelDataptr->MaterialData->NormalMap, uv.x, uv.y);
		NormalMap = make_float3(tmp.x, tmp.y, tmp.z);
		N = UseNormalMap(N, NormalMap, 1.0f);
	}
	float3 DebugColor = make_float3(0);
	float3 RayOrigin = GetPosition();
	float3 radience_indirect = make_float3(0);

	float pdf_diffuse = 0.0f;
	bool IsBxdfRayHitLight=false;
	if (Data.RecursionDepth < params.MaxRecursionDepth) {
		float3 L;
		pdf_diffuse = abs(dot(N, L)) * REVERSE_PI;
		// 先为所有bxdf产生射线方向

		L = ImportanceSampleCosWeight(Data.Seed, N);

		PerRayData DataBxdf;
		DataBxdf.Radience = make_float3(0);
		DataBxdf.RecursionDepth = Data.RecursionDepth;
		DataBxdf.Seed = Data.Seed;

		optixTraceWithPerRayData(DataBxdf, RayOrigin, L, 0, 2, 0);
		IsBxdfRayHitLight = DataBxdf.RayHitType == HIT_TYPE_LIGHT;
		radience_indirect = DataBxdf.Radience * BaseColor;
	}
	float3 radience_direct = make_float3(0);

	float3 SamplePoint = RandomSamplePointOnLight(Data.Seed);
	float3 ray_dir_direct = normalize(SamplePoint - RayOrigin);

	float pdf_light = 1 / params.areaLight.Area;
	if (dot(N_Geo, ray_dir_direct) > 1e-2f && ray_dir_direct.z > 1e-2f) {
		PerRayData DataDirect;
		optixTraceWithPerRayData(DataDirect, RayOrigin, ray_dir_direct, 1, 2, 0);

		float Dw = params.areaLight.Area * saturate(dot(N, ray_dir_direct) + 1e-4f) * saturate(ray_dir_direct.z) / squared_length(RayOrigin - SamplePoint);
		radience_direct = DataDirect.Radience * Dw * BaseColor * REVERSE_PI;
	}
	
	if (IsBxdfRayHitLight) {
		Data.Radience = (radience_direct * pdf_light + radience_indirect * pdf_diffuse) / (pdf_diffuse + pdf_light);
	}
	else {
		Data.Radience = radience_direct + radience_indirect;
	}
	Data.Radience = FilterGlossy(Data.Radience, 10);
	Data.RayHitType = HIT_TYPE_SCENE;
	SetPerRayDataForPayLoad(Data);
}
// 反射,粗糙金属
extern "C" __global__ void __closesthit__glossy()
{
	PerRayData Data = FetchPerRayDataFromPayLoad();
	if (GetModelDataPtr()->MaterialData->MaterialType == MATERIAL_AREALIGHT) {
		Data.Radience = params.areaLight.Color;
		Data.RayHitType = HIT_TYPE_LIGHT;
		SetPerRayDataForPayLoad(Data);
		return;
	}
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
	float3 N_Geo = N;
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
	if (ModelDataptr->MaterialData->NormalMap != NO_TEXTURE_HERE) {
		float4 tmp = SampleTexture2D<float4>(ModelDataptr->MaterialData->NormalMap, uv.x, uv.y);
		NormalMap = make_float3(tmp.x, tmp.y, tmp.z);
		N = UseNormalMap(N, NormalMap, 1.0f);
	}
	float3 DebugColor = make_float3(0);
	float3 RayOrigin = GetPosition();
	float3 radience_indirect = make_float3(0);

	float pdf_spec = 0.0f;
	bool IsBxdfRayHitLight = false;
	if (Data.RecursionDepth < params.MaxRecursionDepth) {
		float3 L;
		// 产生反射方向
		float3 H = ImportanceSampleGGX(Data.Seed, nullptr, GetModelDataPtr()->MaterialData->Roughness);
		float3 T, B;
		{
			GetTBNFromN(N, T, B);
			H = T * H.x + B * H.y + N * H.z;
			H = normalize(H);
		}
		L = normalize(2 * dot(H, V) * H - V);

		float pdf_m = DistributionGGX(abs(dot(N, H)), Roughness) * dot(N, H);
		float jacob_reflection = 1.0f / (4 * abs(dot(H, L)));
		pdf_spec = pdf_m * jacob_reflection;

		PerRayData DataBxdf;
		DataBxdf.Radience = make_float3(0);
		DataBxdf.RecursionDepth = Data.RecursionDepth;
		DataBxdf.Seed = Data.Seed;

		float HdotV = dot(V, H);
		float3 Ctint = normalize(BaseColor);
		float3 Cs = lerp(0.08 * Specular * lerp(make_float3(1), Ctint, SpecularTint), BaseColor, 1);
		float3 Fs = Cs + (1 - Cs) * Pow5(1 - HdotV);
		//法线分布函数
		float Ds = DistributionGGX(N, H, Roughness);
		//遮蔽项
		float Gs = Smith_G(N, H, V, L, Roughness);
		float3 brdf = Fs * Gs * Ds / abs(4 * dot(N, V) * dot(N, L));

		optixTraceWithPerRayData(DataBxdf, RayOrigin, L, 0, 2, 0);
		IsBxdfRayHitLight = DataBxdf.RayHitType == HIT_TYPE_LIGHT;
		radience_indirect = DataBxdf.Radience * abs(dot(N, L)) * brdf / pdf_spec;
	}
	float3 radience_direct = make_float3(0);

	float3 SamplePoint = RandomSamplePointOnLight(Data.Seed);
	float3 ray_dir_direct = normalize(SamplePoint - RayOrigin);

	float pdf_light = 1 / params.areaLight.Area;
	if (dot(N_Geo, ray_dir_direct) > 1e-2f && ray_dir_direct.z > 1e-2f) {
		PerRayData DataDirect;
		optixTraceWithPerRayData(DataDirect, RayOrigin, ray_dir_direct, 1, 2, 0);

		float3 H_d = normalize(V + ray_dir_direct);
		float HdotV = dot(V, H_d);
		float3 Ctint = normalize(BaseColor);
		float3 Cs = lerp(0.08 * Specular * lerp(make_float3(1), Ctint, SpecularTint), BaseColor, 1);
		float3 Fs = Cs + (1 - Cs) * Pow5(1 - HdotV);
		//法线分布函数
		float Ds = DistributionGGX(N, H_d, Roughness);
		//遮蔽项
		float Gs = Smith_G(N, H_d, V, ray_dir_direct, Roughness);
		float3 brdf = Fs * Gs * Ds / abs(4 * dot(N, V) * dot(N, ray_dir_direct));

		float Dw = params.areaLight.Area * saturate(dot(N, ray_dir_direct) + 1e-4f) * saturate(ray_dir_direct.z) / squared_length(RayOrigin - SamplePoint);
		radience_direct = DataDirect.Radience * Dw * brdf;
	}

	if (IsBxdfRayHitLight) {
		Data.Radience = (radience_direct * pdf_light + radience_indirect * pdf_spec) / (pdf_spec + pdf_light);
	}
	else {
		Data.Radience = radience_direct + radience_indirect;
	}
	Data.Radience = FilterGlossy(Data.Radience, 10);
	Data.RayHitType = HIT_TYPE_SCENE;
	SetPerRayDataForPayLoad(Data);
}

// 玻璃，折射
extern "C" __global__ void __closesthit__glass()
{
	PerRayData Data = FetchPerRayDataFromPayLoad();
	if (GetModelDataPtr()->MaterialData->MaterialType == MATERIAL_AREALIGHT) {
		Data.Radience = params.areaLight.Color;
		Data.RayHitType = HIT_TYPE_LIGHT;
		SetPerRayDataForPayLoad(Data);
		return;
	}
	// 使用轮盘赌决定是否停止
	float roulette_ps = params.MaxRecursionDepth / (params.MaxRecursionDepth + 1.0f);
	float rand_for_roulette = Rand(Data.Seed);
	if (rand_for_roulette > roulette_ps) {
		Data.Radience = make_float3(0);
		Data.RayHitType = HIT_TYPE_SCENE;
		SetPerRayDataForPayLoad(Data);
		return;
	}
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
	float3 N_Geo = N;
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
	float3 radience_indirect = make_float3(0);

	// 假设面法线指向空气一侧
	bool in_surface = dot(N, V) >= 0.0f;
	float eta_i = in_surface ? 1.0f : ior;
	float eta_o = in_surface ? ior : 1.0f;
	float3 n_forward = in_surface ? N : -N;
	float3 h_forward;

	float ps_re = 0.0f;
	float ps_tr = 0.0f;
	bool IsBxdfRayHitLight = false;
	float3 radience = make_float3(0);
	if (Data.RecursionDepth < params.MaxRecursionDepth) {
		float3 L;
		// 产生反射方向
		float3 H = ImportanceSampleGGX(Data.Seed, nullptr, GetModelDataPtr()->MaterialData->Roughness);
		float3 T, B;
		{
			GetTBNFromN(N, T, B);
			H = T * H.x + B * H.y + N * H.z;
			H = normalize(H);
		}
		h_forward = in_surface ? H : -H;

		bool internal_reflection = false;
		// 计算菲涅尔，并参考菲涅尔计算发射什么射线
		// 玻璃材质球分为两部分，反射和透射，brdf和btdf
		// 使用MIS同时对两个部分采样，使用映射后的菲涅尔作为每种采样的概率，
		// 反射采样的qs为映射后的菲涅尔，ps为pm*p_re
		// 折射采样的qs为1-映射后的菲涅尔, ps为pm*p_trans
		// 在brdf中不省略NdotL项
		// 发现了之前测试中遇到的问题，当ior为1时，eta_i * dot(V, h_forward) + eta_o * dot(L, h_forward)严格为0，应将ior最小值设置为1.0001f
		// 最小射线长度设置为1e-4，设置为1e-3f时在玻璃的界面会出现问题
		float fs;
		float c = abs(dot(V, h_forward));
		float g = pow2(eta_o / eta_i) - 1.0f + c * c;
		if (g < 0.0f) {
			// 全反射
			fs = 1.0f;
		}
		else {
			g = sqrt(g);
			fs = 0.5f * pow2((g - c) / (g + c)) * (1 + pow2((c * (g + c) - 1) / (c * (g - c) + 1)));
		}
		fs = saturate(fs);
		float qs_re = lerp(0.1, 0.9, fs);
		float qs_tr = 1 - qs_re;

		float rand_num = Rand(Data.Seed);
		bool is_reflect;
		if (rand_num < qs_re) {
			// 反射
			is_reflect = true;
			L = normalize(2 * dot(h_forward, V) * h_forward - V);
		}
		else {
			//折射
			is_reflect = false;
			L = refract(-V, h_forward, eta_i / eta_o, nullptr);
		}

		float pdf_m = DistributionGGX(abs(dot(h_forward, n_forward)), Roughness) * abs(dot(h_forward, n_forward));
		float jacob_re = 1.0f / (4 * abs(dot(H, L)));
		ps_re = pdf_m * jacob_re;
		float jacob_tr= eta_o * eta_o * abs(dot(L, h_forward)) / pow2(eta_i * dot(V, h_forward) + eta_o * dot(L, h_forward));
		ps_tr = pdf_m * jacob_tr;

		PerRayData DataBxdf;
		DataBxdf.Radience = make_float3(0);
		DataBxdf.RecursionDepth = Data.RecursionDepth;
		DataBxdf.Seed = Data.Seed;
		optixTraceWithPerRayData(DataBxdf, RayOrigin, L,1e-4f, 0, 2, 0);

		if (Data.RecursionDepth == 2) {
			Data.DebugData = make_float3(fs);
		}
		else {
			Data.DebugData = DataBxdf.DebugData;
		}
		float HdotV = dot(V, h_forward);
		////法线分布函数
		float Ds = DistributionGGX(h_forward, n_forward, Roughness);
		//遮蔽项
		float Gs = Smith_G(n_forward, h_forward, V, L, Roughness);
		float3 brdf = BaseColor * fs * Gs * Ds / abs(4 * dot(n_forward, V) * dot(n_forward, L));

		float3 numerator = sqrt(BaseColor) * (1 - fs) * Ds * Gs * abs(dot(h_forward, L) * dot(h_forward, V)) * eta_o * eta_o;
		float denominator = abs(dot(n_forward, V) * dot(n_forward, L)) * pow2(eta_i * dot(V, h_forward) + eta_o * dot(L, h_forward));
		float3 btdf = numerator / denominator;
		
		if (is_reflect) {
			radience = DataBxdf.Radience * dot(n_forward, L) * brdf / (ps_re * qs_re);
		}
		else {
			radience = DataBxdf.Radience * abs(dot(n_forward, L)) * btdf / (ps_tr * qs_tr);
		}
	}

	Data.Radience = radience / roulette_ps;
	Data.Radience = FilterGlossy(Data.Radience, 10);
	Data.RayHitType = HIT_TYPE_SCENE;
	SetPerRayDataForPayLoad(Data);
}

extern "C" __global__ void __closesthit__occluded() {
	SbtDataStruct* HitGroupDataPtr = (SbtDataStruct*)optixGetSbtDataPointer();
	ModelData* modeldata_ptr = (ModelData*)HitGroupDataPtr->DataPtr;
	PerRayData Data = FetchPerRayDataFromPayLoad();
	if (modeldata_ptr->MaterialData->MaterialType == MATERIAL_AREALIGHT) {
		Data.Radience = params.areaLight.Color;
		Data.RayHitType = HIT_TYPE_LIGHT;
	}
	else {
		Data.Radience = make_float3(0);
		Data.RayHitType = HIT_TYPE_SCENE;
	}
	SetPerRayDataForPayLoad(Data);
}