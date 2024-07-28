#pragma once
#include "common.cuh"
static __forceinline__ __device__ float3 fresnelSchlick(float cosTheta, float3 F0)
{
	return F0 + (1.0 - F0) * Pow5(saturate(1.0 - cosTheta));
}
static __forceinline__ __device__ float DielectricFresnel(float3 H, float3 Win, float3 Wout, float eta_i, float eta_o) {
	float HoWout = abs(dot(H, Wout));
	float HoWin = abs(dot(H, Win));
	float Rs = (eta_i * HoWout - eta_o * HoWin) / (eta_i * HoWout + eta_o * HoWin);
	float Rp = (eta_i * HoWin - eta_o * HoWout) / (eta_i * HoWin + eta_o * HoWout);
	return 0.5 * (Rs * Rs + Rp * Rp);
}
static __forceinline__ __device__ float DielectricFresnel(float3 H, float3 Win, float eta_i, float eta_o) {
	float c = abs(dot(H, Win));
	float g = pow2(eta_o / eta_i) - 1 + c * c;
	if (g < 0.0f) {
		return 1.0f;
	}
	return 0.5f * pow2((g - c) / (g + c)) * (1 + pow2((c * (g + c) - 1) / (c * (g - c) + 1)));
}
static __forceinline__ __device__ float Disney_FD90(float roughness, float3 H, float3 L) {
	float HoL = abs(dot(H, L));
	return 0.5 + 2 * roughness * HoL * HoL;
}
static __forceinline__ __device__ float Disney_FD(float FD90, float3 N, float3 W) {
	return 1 + (FD90 - 1) * (1 - Pow5(abs(dot(N, W))));
}
static __forceinline__ __device__ float Y(float3 color) {
	return 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;
}
static __forceinline__ __device__ float DistributionGGX(float3 N, float3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = dot(N, H);
	if (NdotH < 0.0f) {
		return 0.0f;
	}
	float NdotH2 = NdotH * NdotH;

	float num = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return num / denom;
}
static __forceinline__ __device__ float DistributionGGX(float NdotH, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH2 = NdotH * NdotH;

	float num = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return num / denom;
}
static __forceinline__ __device__ float SmithG1(float3 wm, float3 v, float alpha) {
	float cosTheta = dot(wm, v);
	float sinTheta = saturate(sqrt(1 - cosTheta * cosTheta));
	float tanTheta = abs(sinTheta / cosTheta);
	if (tanTheta == 0.0f)
		return 1.0f;
	if (abs(cosTheta) <= 1e-4f)
		return 0.0f;
	float root = alpha * tanTheta;
	return 2.0f / (1.0f + sqrt(1.0f + root * root));
}
static __forceinline__ __device__ float Smith_G(float3 n,float3 m, float3 v, float3 l, float roughness) {
	float a = roughness * roughness;
	if (dot(v, m) * dot(v, n) < 0.0f || dot(l, m) * dot(l, n) < 0.0f) {
		return 0.0f;
	}
	return SmithG1(m, v, a) * SmithG1(m, l, a);
}
static __device__ void DisneyBrdf(float3* Out_Spec, float3* Out_diff, float3* Sum,
	float3 N, float3 H, float3 V, float3 L,
	float3 BaseColor, float Roughness, float Metallic, float Specular, float SpecularTint) {


	float HdotV = dot(V, H);
	//首先计算漫反射
	float3 DiffuseTerm;
	float FD90 = 0.5 + 2 * HdotV * HdotV * Roughness;
	DiffuseTerm = BaseColor * REVERSE_PI * (1 + (FD90 - 1) * Pow5(1 - dot(N, V))) * (1 + (FD90 - 1) * Pow5(1 - dot(N, L)));

	//菲涅尔
	float3 Ctint = normalize(BaseColor);
	float3 Cs = lerp(0.08 * Specular * lerp(make_float3(1), Ctint, SpecularTint), BaseColor, Metallic);
	float3 Fs = Cs + (1 - Cs) * Pow5(1 - HdotV);
	//法线分布函数
	float Ds = DistributionGGX(N, H, Roughness);
	//遮蔽项
	float Gs = Smith_G(N,H, V, L, Roughness);
	float3 SpecularTerm = Fs * Gs * Ds / abs(4 * dot(N, V) * dot(N, L));

	if (Out_Spec) {
		Out_Spec[0] = SpecularTerm;
	}
	if (Out_diff) {
		Out_diff[0] = DiffuseTerm;
	}
	if (Sum) {
		Sum[0] = SpecularTerm + (1.0f - Metallic)*DiffuseTerm;
	}
}

__device__ void PrincipledBsdf(uint RecursionDepth,SurfaceData surfaceData,float3 RayDirection,float3& RayDirectionNext,float3& BxdfWeight){
	// 原理话bsdf包含brdf和btdf
	uint3 Id=optixGetLaunchIndex();
	float QsGlass=(1-surfaceData.Metallic)*surfaceData.Transmission;
	float QsDielectric=1-QsGlass;
	float NoiseSeq[9];
	float3 Noise3=hash33(make_uint3(Id.x,Id.y,
		RayTracingGlobalParams.FrameNumber*RayTracingGlobalParams.MaxRecursionDepth+RecursionDepth),
		1103515245U);
	NoiseSeq[0]=Noise3.x;
	NoiseSeq[1]=Noise3.y;
	NoiseSeq[2]=Noise3.z;
	Noise3=hash33(make_uint3(Id.x,Id.y,
		RayTracingGlobalParams.FrameNumber*RayTracingGlobalParams.MaxRecursionDepth+RecursionDepth),
		134775813U);
	NoiseSeq[3]=Noise3.x;
	NoiseSeq[4]=Noise3.y;
	NoiseSeq[5]=Noise3.z;

	float3 V=normalize(-RayDirection);
	bool InSurface=dot(surfaceData.Normal,V)>=0.0f;
	float EtaI=InSurface ? 1:surfaceData.ior;
	float EtaO=InSurface ? surfaceData.ior:1;
	// 与射线方向同向的法线
	float3 NForward=InSurface ? surfaceData.Normal : -surfaceData.Normal;
	float PsReflect=0.0f;
	float PsTransmission=0.0f;
	float Pdf;
	float3 Weight;
	if(NoiseSeq[0]<QsGlass){
		// 使用玻璃材质
		float3 H = ImportanceSampleGGX(make_float2(NoiseSeq[2],NoiseSeq[3]), nullptr, surfaceData.Roughness);
		float3 T, B;
		{
			GetTBNFromN(surfaceData.Normal, T, B);
			H = T * H.x + B * H.y + surfaceData.Normal * H.z;
			H = normalize(H);
		}
		float3 HForward = InSurface ? H : -H;

		// 计算菲涅尔
		float fs;
		{
			float c = abs(dot(V, HForward));
			float g = pow2(EtaO / EtaI) - 1.0f + c * c;
			if (g < 0.0f) {
				// 全反射
				fs = 1.0f;
			}
			else {
				g = sqrt(g);
				fs = 0.5f * pow2((g - c) / (g + c)) * (1 + pow2((c * (g + c) - 1) / (c * (g - c) + 1)));
			}
			fs = saturate(fs);
		}
		float QsReflect=lerp(0.1,0.9,fs);
		float QsTransmission=1-QsReflect;
		float3 L;
		if(NoiseSeq[1]<QsTransmission){
			// 折射
			L = refract(-V, HForward, EtaI / EtaO, nullptr);
		}
		else{
			// 反射
			L = normalize(2 * dot(HForward, V) * HForward - V);
		}
		float PdfM = DistributionGGX(abs(dot(HForward, NForward)), surfaceData.Roughness) * abs(dot(HForward, NForward));
		float JacobReflect = 1.0f / (4 * abs(dot(HForward, L)));
		float PdfReflect = PdfM * JacobReflect;
		float JacobTransmission = EtaO * EtaO * abs(dot(L, HForward)) / pow2(EtaI * dot(V, HForward) + EtaO * dot(L, HForward));
		float PdfTransmission = PdfM * JacobTransmission;

		float Ds = DistributionGGX(HForward, NForward, surfaceData.Roughness);
		//遮蔽项
		float Gs = Smith_G(NForward, HForward, V, L, surfaceData.Roughness);
		float3 Brdf = surfaceData.BaseColor * fs * Gs * Ds / abs(4 * dot(NForward, V) * dot(NForward, L));

		float3 numerator = sqrt(surfaceData.BaseColor) * (1 - fs) * Ds * Gs * abs(dot(HForward, L) * dot(HForward, V)) * EtaO * EtaO;
		float denominator = abs(dot(NForward, V) * dot(NForward, L)) * pow2(EtaI * dot(V, HForward) + EtaO * dot(L, HForward));
		float3 Btdf = numerator / denominator;

		if(NoiseSeq[1]<QsTransmission){
			// 折射
			Weight=saturate(-dot(NForward, L)) * Btdf / (PdfTransmission * QsTransmission);
		}
		else{
			// 反射
			Weight=saturate(dot(NForward, L)) * Brdf / (PdfReflect * QsReflect);
		}
		RayDirectionNext=L;
	}
	else{
		// 不透明材质
		float QsDiffuse=1-surfaceData.Metallic;
		float QsReflect=1;
		float QsSum=QsDiffuse+QsReflect;
		QsDiffuse/=QsSum;
		QsReflect/=QsSum;
		// 根据概率选择漫射或反射
		float3 L;
		if(NoiseSeq[1]<QsDiffuse){
			L=ImportanceSampleCosWeight(make_float2(NoiseSeq[2],NoiseSeq[3]),NForward);
		}
		else{
			float3 H = ImportanceSampleGGX(make_float2(NoiseSeq[4],NoiseSeq[5]), nullptr, surfaceData.Roughness);
			float3 T, B;
			{
				GetTBNFromN(surfaceData.Normal, T, B);
				H = T * H.x + B * H.y + surfaceData.Normal * H.z;
				H = normalize(H);
			}
			float3 HForward = InSurface ? H : -H;
			L = normalize(2 * dot(HForward, V) * HForward - V);
		}
		float3 HForward=normalize(V+L);

		float PdfDiffuse=saturate(dot(NForward,L))*REVERSE_PI;
		float PdfM=DistributionGGX(saturate(dot(NForward, HForward)), surfaceData.Roughness) * abs(dot(NForward, HForward));
		float JacobReflect=1/(4*abs(dot(HForward,L)));
		float PdfReflect=PdfM*JacobReflect;

		float HdotV=abs(dot(V,HForward));
		float3 Ctint = normalize(surfaceData.BaseColor);
		float3 Cs = lerp(0.08 * surfaceData.Specular * lerp(make_float3(1), Ctint, surfaceData.SpecularTint), surfaceData.BaseColor, surfaceData.Metallic);
		float3 Fs = Cs + (1 - Cs) * Pow5(1 - HdotV);
		//法线分布函数
		float Ds = DistributionGGX(saturate(dot(NForward, HForward)), surfaceData.Roughness);
		//遮蔽项
		float Gs = Smith_G(NForward, HForward, V, L, surfaceData.Roughness);
		float3 BrdfSpecular = Fs * Gs * Ds / abs(4 * dot(NForward, V) * dot(NForward, L));
		float3 BrdfDiffuse = surfaceData.BaseColor * REVERSE_PI;
		float3 Brdf = (1 - surfaceData.Metallic) * BrdfDiffuse + BrdfSpecular;
		Pdf=PdfReflect*QsReflect+PdfDiffuse*QsDiffuse;
		Weight=saturate(dot(NForward,L))*Brdf/Pdf;
		RayDirectionNext=L;
	}
	BxdfWeight=Weight;
}