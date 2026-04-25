#pragma once
#include "common.cuh"
static DEVICE float3 ImportanceSampleCosWeight(float2 rand) {
	float r = sqrt(rand.x);
	float phi = rand.y * 2.0f * PI;
	float cos_phi = cos(phi);
	float sin_phi = sin(phi);
	float3 RayDir = make_float3(r * cos(phi), r * sin(phi), sqrt(fmaxf(0.0f, 1 - r * r)));
	return ASSERT_VALID(RayDir);
}

static DEVICE float3 ImportanceSampleCosWeight(float2 rand, float3 N) {
	float3 RayDir = ImportanceSampleCosWeight(rand);
	float3 T, B;
	GetTBNFromN(N, T, B);
	RayDir = normalize(T * RayDir.x + B * RayDir.y + N * RayDir.z);
	return ASSERT_VALID(RayDir);
}

static DEVICE float3 ImportanceSampleGGX(float2 Xi, float roughness)
{
	//if (roughness < 5e-2f) {
	//	return make_float3(0, 0, 1);
	//}
	//Xi.y = fminf(Xi.y, 0.99f);
	//float a = roughness * roughness;
	//float phi = 2.0 * PI * Xi.x;
	//float numerator = (1.0 - Xi.y);
	//float denominator = (1.0 + (a * a - 1.0) * Xi.y);
	//float cosTheta = sqrt(numerator / denominator);
	//ASSERT_VALID(cosTheta);
	//float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	//ASSERT_VALID(sinTheta);
	//// from spherical coordinates to cartesian coordinates
	//float3 H;
	//H.x = cos(phi) * sinTheta;
	//H.y = sin(phi) * sinTheta;
	//H.z = cosTheta;
	//return ASSERT_VALID(H);

	float a = roughness * roughness;
	if (roughness < 1e-4f) return make_float3(0, 0, 1);
	float3 ray = ImportanceSampleCosWeight(Xi);
	ray = ray * make_float3(a, a, 1);
	float len2 = dot(ray, ray);
	if (len2 < 1e-8) return make_float3(0, 0, 1);
	return ray / sqrt(len2);

}

static DEVICE float3 ImportanceSampleGGX(float2 noise, float roughness, float3 N) {
	float3 H = ImportanceSampleGGX(noise, roughness);
	float3 T, B;
	GetTBNFromN(N, T, B);
	H = normalize(T * H.x + B * H.y + N * H.z);
	return ASSERT_VALID(H);
}
 
 
static INLINE DEVICE float3 fresnelSchlick(float cosTheta, float3 F0)
{
	return F0 + (1.0 - F0) * Pow5(saturate(1.0 - cosTheta));
}
static INLINE DEVICE float DielectricFresnel(float3 HForward, float3 V, float EtaI, float EtaO) {
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
	return ASSERT_VALID(fs);
}
static INLINE DEVICE float Disney_FD90(float roughness, float3 H, float3 L) {
	float HoL = abs(dot(H, L));
	return 0.5 + 2 * roughness * HoL * HoL;
}
static INLINE DEVICE float Disney_FD(float FD90, float3 N, float3 W) {
	return 1 + (FD90 - 1) * (1 - Pow5(abs(dot(N, W))));
}
static INLINE DEVICE float Y(float3 color) {
	return 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;
}

static INLINE DEVICE float DistributionGGX(float NdotH, float roughness)
{
	float a = roughness* roughness;
	if (NdotH < 0.0f) {
		return 0.0f;
	}
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
	
	float num = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;
	ASSERT_VALID(num / fmaxf(denom, FloatEpsilon));
	return ASSERT_VALID(num / fmaxf(denom, FloatEpsilon));
}
static INLINE DEVICE float DistributionGGX(float3 N, float3 H, float roughness)
{
	return DistributionGGX(dot(N, H), roughness);
}
static INLINE DEVICE float SmithG1(float3 wm, float3 v, float alpha) {
	float cosTheta = dot(wm, v);
	float sinTheta = saturate(sqrt(1 - cosTheta * cosTheta));
	float tanTheta = abs(sinTheta / cosTheta);
	if (tanTheta == 0.0f)
		return 1.0f;
	if (abs(cosTheta) <= 1e-4f)
		return 0.0f;
	float root = alpha * tanTheta;
	return ASSERT_VALID(2.0f / (1.0f + sqrt(1.0f + root * root)));
}
static INLINE DEVICE float Smith_G(float3 n,float3 m, float3 v, float3 l, float roughness) {
	float a = roughness * roughness;
	if (dot(v, m) * dot(v, n) < 0.0f || dot(l, m) * dot(l, n) < 0.0f) {
		return 0.0f;
	}
	return ASSERT_VALID(SmithG1(m, v, a) * SmithG1(m, l, a));
}
static DEVICE float3 SpecularBrdf(SurfaceData& surfaceData,
	float3 NForward, float3 HForward, float3 V, float3 L,float EtaI,float EtaO) {
	if (!(dot(NForward, V) > 0.0f && dot(NForward, L) > 0.0f)) {
		return make_float3(0);
	}
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
	return ASSERT_VALID(brdf);
}
static DEVICE float3 DiffuseBrdf(SurfaceData& data) {
	//首先计算漫反射
	float3 DiffuseTerm;
	DiffuseTerm = data.BaseColor * REVERSE_PI;
	return (1 - data.Transmission) * (1 - data.Metallic) * DiffuseTerm;
}
static DEVICE float3 TransmissionBtdf(SurfaceData& surfaceData,
	float3 NForward, float3 HForward, float3 V, float3 L,float EtaO,float EtaI) {
	if (!(dot(NForward, V) > 0.0f && dot(NForward, L) < 0.0f)) {
		return make_float3(0);
	}
	float fs = DielectricFresnel(HForward, V, EtaI, EtaO);
	float Ds = DistributionGGX(HForward, NForward, surfaceData.Roughness);
	//遮蔽项
	float Gs = Smith_G(NForward, HForward, V, L, surfaceData.Roughness);

	float3 numerator = sqrt(surfaceData.BaseColor) * (1 - fs) * Ds * Gs * abs(dot(HForward, L) * dot(HForward, V)) * EtaO * EtaO;
	float denominator = abs(dot(NForward, V) * dot(NForward, L)) * pow2(EtaI * dot(V, HForward) + EtaO * dot(L, HForward));
	return ASSERT_VALID((surfaceData.Transmission) * (1 - surfaceData.Metallic) * numerator / fmaxf(denominator, FloatEpsilon));
}

DEVICE float3 SampleBsdf(SurfaceData& surfaceData,float3 noise,float3 V,bool& IsTransmission,float3& H) {
	bool InSurface = dot(surfaceData.VertexNormal, V) >= 0.0f;
	float EtaI = InSurface ? 1 : surfaceData.ior;
	float EtaO = InSurface ? surfaceData.ior : 1;
	// 与射线方向同向的法线
	float3 NForward = InSurface ? surfaceData.Normal : -surfaceData.Normal;

	float3 HForward = ASSERT_VALID(ImportanceSampleGGX(make_float2(noise.x, noise.y), surfaceData.Roughness, NForward));
	float QReflect = 1;
	float QDiffuse = (1 - surfaceData.Metallic) * (1 - surfaceData.Transmission);
	float QTransmission = (1 - surfaceData.Metallic) * surfaceData.Transmission * (1 - DielectricFresnel(HForward, V, EtaI, EtaO));
	float QSum = QReflect + QDiffuse + QTransmission;
	QReflect /= QSum;
	QTransmission /= QSum;
	QDiffuse /= QSum;
	if (noise.z < SAFETY_MARGIN(QTransmission)) {
		float3 L = ASSERT_VALID(refract(-V, HForward, EtaI / EtaO, nullptr));
		IsTransmission = true;
		H = ASSERT_VALID(HForward);
		return ASSERT_VALID(saturateRay(normalize(L)));
	}
	else {
		float3 L;
		IsTransmission = false;
		if (noise.z < SAFETY_MARGIN(QTransmission + QReflect)) {
			L = ASSERT_VALID(normalize(2 * dot(HForward, V) * HForward - V));
			H = HForward;
		}
		else {
			L = ASSERT_VALID(ImportanceSampleCosWeight(make_float2(noise.x, noise.y), NForward));
			H = ASSERT_VALID(normalize(V + L));
		}
		L = ASSERT_VALID(ClampRayDir(L, InSurface ? surfaceData.VertexNormal : -surfaceData.VertexNormal));
		return ASSERT_VALID(saturateRay(normalize(L)));
	}
}
DEVICE float EvalPdf(SurfaceData& surfaceData, float3 V, float3 L,bool IsTransmission,float3 HForward) {
	bool InSurface = dot(surfaceData.VertexNormal, V) >= 0.0f;
	float EtaI = InSurface ? 1 : surfaceData.ior;
	float EtaO = InSurface ? surfaceData.ior : 1;
	// 与射线方向同向的法线surfaceData.Normal
	float3 NForward = InSurface ? surfaceData.Normal : -surfaceData.Normal;

	
	float QReflect = 1;
	float QDiffuse = (1 - surfaceData.Metallic) * (1 - surfaceData.Transmission);
	float QTransmission = (1 - surfaceData.Metallic) * surfaceData.Transmission * (1 - DielectricFresnel(HForward, V, EtaI, EtaO));
	float QSum = QReflect + QDiffuse + QTransmission;
	QReflect /= QSum;
	QTransmission /= QSum;
	QDiffuse /= QSum;

	float PdfM = DistributionGGX(HForward, NForward, surfaceData.Roughness) * abs(dot(HForward, NForward));
	if (IsTransmission) {
		float JacobTransmission = EtaO * EtaO * abs(dot(L, HForward)) / pow2(EtaI * dot(V, HForward) + EtaO * dot(L, HForward));
		float PdfTransmission = PdfM * JacobTransmission;
		return ASSERT_VALID(PdfTransmission * QTransmission);
	}
	else {
		float JacobReflect = 1 / fmaxf(4 * abs(dot(HForward, L)), FloatEpsilon);
		float PdfReflect = PdfM * JacobReflect;
		float PdfCosWeighted = saturate(dot(NForward, L)) * REVERSE_PI;
		return ASSERT_VALID(fmaxf(PdfReflect * QReflect + PdfCosWeighted * QDiffuse, FloatEpsilon));
	}
}

DEVICE float3 EvalBsdf(SurfaceData& surfaceData, float3 V, float3 L, bool IsTransmission, float3 HForward) {
	bool InSurface = dot(surfaceData.VertexNormal, V) >= 0.0f;
	float EtaI = InSurface ? 1 : surfaceData.ior;
	float EtaO = InSurface ? surfaceData.ior : 1;
	// 与V方向同向的法线
	float3 NForward = InSurface ? surfaceData.Normal : -surfaceData.Normal;
	//float QReflect = 1;
	//float QDiffuse = (1 - surfaceData.Metallic) * (1 - surfaceData.Transmission);
	//float QTransmission = (1 - surfaceData.Metallic) * surfaceData.Transmission * (1 - DielectricFresnel(HForward, V, EtaI, EtaO));
	//float QSum = QReflect + QDiffuse + QTransmission;
	//QReflect /= QSum;
	//QTransmission /= QSum;
	//QDiffuse /= QSum;

	float PdfM = DistributionGGX(HForward, NForward, surfaceData.Roughness) * abs(dot(HForward, NForward));
	if (IsTransmission) {
		return ASSERT_VALID(dot(NForward, L) < 0.0f ? TransmissionBtdf(surfaceData, NForward, HForward, V, L, EtaO, EtaI) * saturate(-dot(NForward, L)) : make_float3(0));
	}
	else {
		return ASSERT_VALID(dot(NForward, L) < 0.0f ? make_float3(0) : (DiffuseBrdf(surfaceData) + SpecularBrdf(surfaceData, NForward, HForward, V, L, EtaI, EtaO)) * saturate(dot(NForward, L)));
	}
}

DEVICE float3 PrincipledBsdf(uint RecursionDepth, SurfaceData surfaceData,float3 Noise3, float3 V, float3& BxdfWeight, bool& IsTransmission) {
	float3 HForward;
	float3 L = SampleBsdf(surfaceData, Noise3, V, IsTransmission, HForward);
	float pdf = EvalPdf(surfaceData, V, L, IsTransmission, HForward);
	float3 Bsdf = EvalBsdf(surfaceData, V, L, IsTransmission, HForward);
	BxdfWeight = Bsdf / pdf;
	return L;
}

DEVICE INLINE bool IsRayContributeToBtdf(float3 RayDir,float3 VertexNormal,float3 V) {
	bool InSurface = dot(VertexNormal, V) >= 0.0f;
	// 与V方向同向的法线
	float3 NForward = InSurface ? VertexNormal : -VertexNormal;
	if (dot(NForward, RayDir) >= 0) {
		return false;
	}
	else {
		return true;
	}
}