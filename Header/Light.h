#pragma once
#include "common.h"

enum LightType :uint {
	Sphere = 0xFFFF0000,
	Area = 0xFFFF0001,
	Directional = 0xFFFF0002
};

// 编码规则，0是序号，1是类型，234是颜色
class SphereLight {
private:
	float3 pos;
	float radius;
	float3 color;
public:
	SphereLight() = delete;
	SphereLight(float3 Pos, float Radius,float3 Color);
	SphereLight(float3 Pos, float Radius, float3 Color,float intensity);
	OptixAabb GetAabb();
	ProceduralGeometryMaterialBuffer PackMaterialBuffer();
};

class RectangleLight {
private:
	float3 p1, p2, p3, p4;
	float3 color;
	float max4(float a, float b, float c, float d);
	float min4(float a, float b, float c, float d);
	inline static float length(float3 v) {
		return std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	}
public:
	RectangleLight() = delete;
	RectangleLight(float3 p1, float3 p2, float3 p3, float3 p4, float3 Color, float intensity);
	OptixAabb GetAabb();
	ProceduralGeometryMaterialBuffer PackMaterialBuffer();

};

class LightManager {
public:
	static inline LightManager& GetInstance() {
		static LightManager m;
		return m;
	}
	inline void Add(CUdeviceptr p) {
		materialArrayPtr.push_back(p);
	}
	inline uint GetNewLightIndex() {
		return materialArrayPtr.size();
	}
	inline void UploadLightList() {
		if (gpuArray) {
			throw std::runtime_error("Light manager has been already uploaded");
		}
		else {
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuArray), sizeof(CUdeviceptr) * materialArrayPtr.size()));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(gpuArray), materialArrayPtr.data(), sizeof(CUdeviceptr) * materialArrayPtr.size(), cudaMemcpyHostToDevice));
		}
	}
	inline uint GetLength() {
		return materialArrayPtr.size();
	}
	inline CUdeviceptr GetPtr() {
		return gpuArray;
	}
private:
	LightManager() = default;
	vector<CUdeviceptr> materialArrayPtr;
	CUdeviceptr gpuArray= 0;
	~LightManager() {
		if (gpuArray) {
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(gpuArray)));
		}
	}
	LightManager(const LightManager&) = delete;
	LightManager& operator=(const LightManager&) = delete;
};