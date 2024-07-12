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