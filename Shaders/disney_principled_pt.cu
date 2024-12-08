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
extern "C" __global__ void __closesthit__principled_bsdf()
{
	// 迪士尼的原理化bsdf
	// 由三种材质构成，玻璃，电介质和金属
	// 分别发射三种射线，漫射射线、反射射线、折射射线
	// 一次只发射一种射线，使用mis进行采样
	// 根据透射和金属度决定三种材质的混合比例
	// 漫射光线和反射光线贡献brdf，折射光线贡献btdf
	// 对于上半球面，漫射和反射光线的pdf不为0，对于下半球面，折射光线的pdf不为0
	PerRayData Data = FetchPerRayDataFromPayLoad();
	if (GetModelDataPtr()->MaterialData->MaterialType == MATERIAL_AREALIGHT) {
		Data.Radience = RayTracingGlobalParams.areaLight.Color;
		Data.RayHitType = HIT_TYPE_LIGHT;
		SetPerRayDataForPayLoad(Data);
		return;
	}
	const uint3 Id = optixGetLaunchIndex();
	// 使用轮盘赌决定是否停止
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
	// 计算几何法线
	float3 N_Geo;
	{
		float3 v1, v2, v3;
		GetTriangle(v1, v2, v3);
		N_Geo = normalize(cross(v1 - v2, v1 - v3));
	}
	if (IsTextureViewValid(ModelDataptr->MaterialData->BaseColorMap)) {
		float4 tmp = SampleTexture2DRuntimeSpecific(ModelDataptr->MaterialData->BaseColorMap, uv.x, uv.y);
		BaseColor = make_float3(tmp.x, tmp.y, tmp.z);
	}
	else {
		BaseColor = ModelDataptr->MaterialData->BaseColor;
	}
	BaseColor *= AO;
	if (IsTextureViewValid(ModelDataptr->MaterialData->ARMMap)) {
		float4 tmp = SampleTexture2DRuntimeSpecific(ModelDataptr->MaterialData->ARMMap, uv.x, uv.y);
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
	if (IsTextureViewValid(ModelDataptr->MaterialData->NormalMap)) {
		float4 tmp = SampleTexture2DRuntimeSpecific(ModelDataptr->MaterialData->NormalMap, uv.x, uv.y);
		NormalMap = make_float3(tmp.x, tmp.y, tmp.z);
		N = UseNormalMap(N, NormalMap, 1.0f);
	}
	float3 DebugColor = make_float3(0);
	float3 RayOrigin = GetPosition();

	// 假设面法线指向空气一侧
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
	// 电介质 | 金属 | 玻璃
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
		// 处理间接光部分
		switch (MatToTrace)
		{
		case Dielectric: {
			float qs_diffuse = 1 - Metallic;
			float qs_re = 1;
			float qs_sum = qs_diffuse + qs_re;
			qs_diffuse /= qs_sum;
			qs_re /= qs_sum;
			// 电介质有镜射和漫射
			if (rand_for_select_ray < qs_diffuse) {
				// 漫射
				L = ImportanceSampleCosWeight(Data.Seed, N);//??????
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
			// 限制L在指向外侧
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
			//法线分布函数
			float Ds = DistributionGGX(abs(dot(n_forward, H)), Roughness);
			//遮蔽项
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

			float rand_num = rand_for_select_ray;
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
			float jacob_tr = eta_o * eta_o * abs(dot(L, h_forward)) / pow2(eta_i * dot(V, h_forward) + eta_o * dot(L, h_forward));
			ps_tr = pdf_m * jacob_tr;

			PerRayData DataBxdf;
			DataBxdf.Radience = make_float3(0);
			DataBxdf.RecursionDepth = Data.RecursionDepth;
			DataBxdf.Seed = Data.Seed;
			optixTraceWithPerRayData(DataBxdf, RayOrigin, L, 1e-4f, 0, 2, 0);
			////法线分布函数
			float Ds = DistributionGGX(h_forward, n_forward, Roughness);
			//遮蔽项
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
	
	// 直接光
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
			//法线分布函数
			float Ds = DistributionGGX(n_forward, H_d, Roughness);
			//遮蔽项
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
		RayTracingGlobalParams.IndirectOutputBuffer[pixel_id] = RayTracingGlobalParams.areaLight.Color;
		return;
	}
	for(;RecursionDepth<RayTracingGlobalParams.MaxRecursionDepth;RecursionDepth++){
		// 这次的着色点由上一次追踪给出
		// 判断是否miss
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
			Radience+=Weight*RayTracingGlobalParams.areaLight.Color;
			break;
		}
		// 加载命中点
		SurfaceData surfaceData;
		surfaceData.Clear();
		surfaceData.Load(hitInfo);
		
		// 假设只考虑漫射
		bool TraceGlass;
		float4 Noise4=hash44(make_uint4(idx.x,idx.y,RayTracingGlobalParams.FrameNumber,RecursionDepth));
		float3 BxdfWeight;
		PrincipledBsdf(RecursionDepth,surfaceData,RayDirection,BxdfWeight,TraceGlass);
		// 立即追踪bxdf光线
		TraceRay(hitInfo,surfaceData.Position, RayDirection,1e-3f, 0, 1, 0);
		// 直接光
		float3 Color, LightCenter;
		float Radius;
		float* sphereLightData=FetchLightData(0);
		DecodeSphereLight(sphereLightData, LightCenter, Radius, Color);
		float4 SampleResult = SampleSphereLight(Noise4.w, Noise4.z, surfaceData.Position, LightCenter, Radius);
		float3 RayDirDirectLight = make_float3(SampleResult.x, SampleResult.y, SampleResult.z);
		float pdfSphereLight = SampleResult.w;
		float3 RadienceDirect=make_float3(0.0f);
		if(!TraceGlass){
			HitInfo hitInfoDirectLight;
			TraceRay(hitInfoDirectLight, RayOrigin, RayDirDirectLight,1e-3f, 0, 1, 0);
			float3 lightColor=hitInfoDirectLight.surfaceType==SurfaceType::Light ? Color : make_float3(0.0f);
			float Dw = RayTracingGlobalParams.areaLight.Area * saturate(dot(surfaceData.Normal, RayDirDirectLight) + 1e-4f);
			RadienceDirect = lightColor * Dw * surfaceData.BaseColor * REVERSE_PI;
			// MIS
			float PdfDiffuse=saturate(dot(surfaceData.Normal,RayDirection))*REVERSE_PI;
			float PdfLight = pdfSphereLight;
			float MISWeight=1.0f;
			if (hitInfo.surfaceType==Light) {
				MISWeight=PdfDiffuse / (PdfDiffuse + PdfLight);
				RadienceDirect*=(PdfLight) / (PdfDiffuse + PdfLight);
			}
			Radience+=Weight*RadienceDirect;
			Weight*=BxdfWeight*MISWeight;
		}
		else{
			Weight*=BxdfWeight;
		}
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
	ProceduralGeometryMaterialBuffer* data = (ProceduralGeometryMaterialBuffer*)(((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr);
	float3 pos;
	float radius;
	pos.x = data->Elements[1];
	pos.y = data->Elements[2];
	pos.z = data->Elements[3];
	radius = data->Elements[4];
    float3& sphere_center = pos;
    float& sphere_radius = radius;
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
	hitInfo.PrimitiveID=0;
	hitInfo.SbtDataPtr=((SbtDataStruct*)optixGetSbtDataPointer())->DataPtr;
	hitInfo.TriangleCentroidCoord=make_float2(0,0);
	hitInfo.surfaceType= SurfaceType::Light;
	SetPayLoad(hitInfo);
}