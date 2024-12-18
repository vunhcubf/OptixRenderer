#pragma once
#include "common.cuh"
static __forceinline__ __device__ float3 fresnelSchlick(float cosTheta, float3 F0)
{
	return F0 + (1.0 - F0) * Pow5(saturate(1.0 - cosTheta));
}
static __forceinline__ __device__ float DielectricFresnel(float3 HForward, float3 V, float EtaI, float EtaO) {
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
	return fs;
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

static __forceinline__ __device__ float DistributionGGX(float NdotH, float roughness)
{
	float a = roughness;
	if (NdotH < 0.01f) {
		float deltaX = 1 - NdotH;
		float deltaR = a;
		deltaX = fmaxf(deltaX, 1e-9f);
		deltaR = fmaxf(deltaR, 1e-9f);
		float inside = 1 / (2 * deltaX / deltaR + deltaR);
		return REVERSE_PI * inside * inside;
	}
	else if (NdotH < 1e-7f) {
		float deltaR = a;
		deltaR = fmaxf(deltaR, 1e-9f);
		float inside = 1 / deltaR;
		return REVERSE_PI * inside * inside;
	}
	float a2 = a * a;
	a2 = fmaxf(a2, FloatEpsilon);
	float NdotH2 = NdotH * NdotH;
	if (NdotH < 0.0f) {
		return 0.0f;
	}
	float num = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;
	ASSERT_VALID(num / fmaxf(denom, FloatEpsilon));
	return num / fmaxf(denom, FloatEpsilon);
}
static __forceinline__ __device__ float DistributionGGX(float3 N, float3 H, float roughness)
{
	return DistributionGGX(dot(N, H), roughness);
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
static __device__ float3 SpecularBrdf(SurfaceData& surfaceData,
	float3 NForward, float3 HForward, float3 V, float3 L,float EtaI,float EtaO) {
	float HdotV = dot(V, HForward);
	//菲涅尔
	float3 Ctint = normalize(surfaceData.BaseColor);
	float3 Cs = lerp(0.08 * surfaceData.Specular * lerp(make_float3(1), Ctint, surfaceData.SpecularTint), surfaceData.BaseColor, surfaceData.Metallic);
	float3 Fs = Cs + (1 - Cs) * Pow5(1 - HdotV);
	float3 fs = make_float3(DielectricFresnel(HForward, V, EtaI, EtaO));
	//法线分布函数
	float Ds = DistributionGGX(HForward, NForward, surfaceData.Roughness);
	//遮蔽项
	float Gs = Smith_G(NForward, HForward, V, L, surfaceData.Roughness);
	float3 brdf= lerp(fs, Fs, surfaceData.Metallic) * Gs * Ds / fmaxf(abs(4 * dot(NForward, V) * dot(NForward, L)), FloatEpsilon);
	ASSERT_VALID(brdf);
	return brdf;
}
static __device__ float3 DiffuseBrdf(SurfaceData& data) {
	//首先计算漫反射
	float3 DiffuseTerm;
	DiffuseTerm = data.BaseColor * REVERSE_PI;
	return (1 - data.Transmission) * (1 - data.Metallic) * DiffuseTerm;
}
static __device__ float3 TransmissionBtdf(SurfaceData& surfaceData,
	float3 NForward, float3 HForward, float3 V, float3 L,float EtaO,float EtaI) {
	float fs = DielectricFresnel(HForward, V, EtaI, EtaO);
	float Ds = DistributionGGX(HForward, NForward, surfaceData.Roughness);
	//遮蔽项
	float Gs = Smith_G(NForward, HForward, V, L, surfaceData.Roughness);

	float3 numerator = sqrt(surfaceData.BaseColor) * (1 - fs) * Ds * Gs * abs(dot(HForward, L) * dot(HForward, V)) * EtaO * EtaO;
	float denominator = abs(dot(NForward, V) * dot(NForward, L)) * pow2(EtaI * dot(V, HForward) + EtaO * dot(L, HForward));
	ASSERT_VALID(numerator / fmaxf(denominator, FloatEpsilon));
	return (surfaceData.Transmission) * (1 - surfaceData.Metallic)*numerator / fmaxf(denominator, FloatEpsilon);
}
__device__ void Principle1dBsdf(uint RecursionDepth, SurfaceData surfaceData, float3& RayDirection, float3& BxdfWeight,bool IsTransmission) {
	// 原理话bsdf包含brdf和btdf
	uint3 Id = optixGetLaunchIndex();
	float3 Noise3 = hash33(make_uint3(Id.x, Id.y,
		RayTracingGlobalParams.FrameNumber * RayTracingGlobalParams.MaxRecursionDepth + RecursionDepth),
		1103515245U);
	float3 V = normalize(-RayDirection);
	bool InSurface = dot(surfaceData.Normal, V) >= 0.0f;
	float EtaI = InSurface ? 1 : surfaceData.ior;
	float EtaO = InSurface ? surfaceData.ior : 1;
	// 与射线方向同向的法线
	float3 NForward = InSurface ? surfaceData.Normal : -surfaceData.Normal;
	float Pdf;
	float3 Weight;

	float3 HForward = ImportanceSampleGGX(make_float2(Noise3.x, Noise3.y), surfaceData.Roughness, NForward);
	float QReflect = 1;
	float QDiffuse = (1 - surfaceData.Metallic) * (1 - surfaceData.Transmission);
	float QTransmission = (1 - surfaceData.Metallic) * surfaceData.Transmission * (1 - DielectricFresnel(HForward, V, EtaI, EtaO));
	float QSum = QReflect + QDiffuse + QTransmission;
	QReflect /= QSum;
	QTransmission /= QSum;
	QDiffuse /= QSum;

	float PdfM = DistributionGGX(HForward, NForward, surfaceData.Roughness) * abs(dot(HForward, NForward));
	if (Noise3.z < SAFETY_MARGIN(QTransmission)) {
		float3 L = refract(-V, HForward, EtaI / EtaO, nullptr);
		float JacobTransmission = EtaO * EtaO * abs(dot(L, HForward)) / pow2(EtaI * dot(V, HForward) + EtaO * dot(L, HForward));
		float PdfTransmission = PdfM * JacobTransmission;
		float3 Btdf = TransmissionBtdf(surfaceData, NForward, HForward, V, L, EtaO, EtaI);
		Weight = saturate(-dot(NForward, L)) * Btdf / fmaxf((PdfTransmission * QTransmission), FloatEpsilon);
		ASSERT_VALID(Weight);
		RayDirection = L;
	}
	else if(Noise3.z < SAFETY_MARGIN(QTransmission+QReflect)){
		float3 L = normalize(2 * dot(HForward, V) * HForward - V);
		float JacobReflect = 1 / fmaxf(4 * abs(dot(HForward, L)), FloatEpsilon);
		float PdfReflect = PdfM * JacobReflect;
		float3 BrdfSpecular = SpecularBrdf(surfaceData, NForward, HForward, V, L, EtaI, EtaO);
		Weight = saturate(dot(NForward, L)) * BrdfSpecular / fmaxf(PdfReflect * QReflect, FloatEpsilon);
		ASSERT_VALID(Weight);
		RayDirection = L;
	}
	else {
		float3 L = ImportanceSampleCosWeight(make_float2(Noise3.x,Noise3.y), NForward);
		float PdfDiffuse = saturate(dot(NForward, L)) * REVERSE_PI;
		float3 BrdfDiffuse = DiffuseBrdf(surfaceData);
		Weight = saturate(dot(NForward, L)) * BrdfDiffuse / fmaxf(PdfDiffuse * QDiffuse, FloatEpsilon);
		ASSERT_VALID(Weight);
		RayDirection = L;
	}
	BxdfWeight = Weight;
}

__device__ void PrincipledBsdf(uint RecursionDepth, SurfaceData surfaceData, float3& RayDirection, float3& BxdfWeight, bool IsTransmission) {
	// 原理话bsdf包含brdf和btdf
	uint3 Id = optixGetLaunchIndex();
	float3 Noise3 = hash33(make_uint3(Id.x, Id.y,
		RayTracingGlobalParams.FrameNumber * RayTracingGlobalParams.MaxRecursionDepth + RecursionDepth),
		1103515245U);
	float3 V = normalize(-RayDirection);
	bool InSurface = dot(surfaceData.Normal, V) >= 0.0f;
	float EtaI = InSurface ? 1 : surfaceData.ior;
	float EtaO = InSurface ? surfaceData.ior : 1;
	// 与射线方向同向的法线
	float3 NForward = InSurface ? surfaceData.Normal : -surfaceData.Normal;
	float Pdf;
	float3 Weight;

	float3 HForward = ImportanceSampleGGX(make_float2(Noise3.x, Noise3.y), surfaceData.Roughness, NForward);
	float QReflect = 1;
	float QDiffuse = (1 - surfaceData.Metallic) * (1 - surfaceData.Transmission);
	float QTransmission = (1 - surfaceData.Metallic) * surfaceData.Transmission * (1 - DielectricFresnel(HForward, V, EtaI, EtaO));
	float QSum = QReflect + QDiffuse + QTransmission;
	QReflect /= QSum;
	QTransmission /= QSum;
	QDiffuse /= QSum;

	float PdfM = DistributionGGX(HForward, NForward, surfaceData.Roughness) * abs(dot(HForward, NForward));
	if (Noise3.z < SAFETY_MARGIN(QTransmission)) {
		float3 L = refract(-V, HForward, EtaI / EtaO, nullptr);
		float JacobTransmission = EtaO * EtaO * abs(dot(L, HForward)) / pow2(EtaI * dot(V, HForward) + EtaO * dot(L, HForward));
		float PdfTransmission = PdfM * JacobTransmission;
		float3 Btdf = TransmissionBtdf(surfaceData, NForward, HForward, V, L, EtaO, EtaI);
		Weight = saturate(-dot(NForward, L)) * Btdf / fmaxf((PdfTransmission * QTransmission), FloatEpsilon);
		ASSERT_VALID(Weight);
		RayDirection = L;
	}
	else {
		bool IsReflection;
		float3 L;
		if (Noise3.z < SAFETY_MARGIN(QTransmission + QReflect)) {
			L = normalize(2 * dot(HForward, V) * HForward - V);
			RayDirection = L;
			IsReflection = true;
		}
		else {
			L = ImportanceSampleCosWeight(make_float2(Noise3.x, Noise3.y), NForward);
			RayDirection = L;
			IsReflection = false;
		}
		float3 HNew = IsReflection ? HForward : normalize(V + L);
		float JacobReflect = 1 / fmaxf(4 * abs(dot(HNew, L)),-1);
		float PdfMNew = DistributionGGX(dot(HNew, NForward), surfaceData.Roughness) * abs(dot(HNew, NForward));
		float PdfReflect = PdfMNew * JacobReflect;
		float3 BrdfSpecular = SpecularBrdf(surfaceData, NForward, HNew, V, L, EtaI, EtaO);
		float PdfCosWeighted = saturate(dot(NForward, L)) * REVERSE_PI;
		float3 BrdfDiffuse = DiffuseBrdf(surfaceData);
		Weight = saturate(dot(NForward, L)) * (BrdfSpecular+BrdfDiffuse) / fmaxf(PdfCosWeighted * QDiffuse + PdfReflect * QReflect,FloatEpsilon);
		ASSERT_VALID(Weight);
	}
	BxdfWeight = Weight;
}