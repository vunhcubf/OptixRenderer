#pragma once
#include "render.h"
#include <optix_stubs.h>
#include <chrono>
#include <stdlib.h>
#include <ctime>
#include <glfw3.h>
#include <glad.h>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <cuda_gl_interop.h>

#include "Light.h"
#include "ComputeShader.h"
#include "Texture.h"
#include "objloader.h"

enum INPUT_TYPE {
	W,
	A,
	S,
	D,
	E,
	Q,
	MOUSE_LEFT,
	MOUSE_RIGHT
};
inline bool GetState(uint BitMask, INPUT_TYPE type) {
	return (BitMask >> type) & 1U;
}
inline void SetState(uint& BitMask, INPUT_TYPE type, bool State) {
	if (State) {
		BitMask |= (1U << type);  // 使用 OR 操作来设置位为 1
	}
	else {
		BitMask &= ~(1U << type); // 使用 AND 操作来设置位为 0
	}
}

class MyCamera {
	float3 WorldPos;
	float Fov = 50/ Deg2Rad;
	static constexpr float Deg2Rad = 57.2957795131f;
	float mPhi = 1.0355;
	float mTheta = 0.756677;
	float3 ForwardDirection;
	float3 UpDirection;
	float3 RightDirection;
public:
	MyCamera(float3 Pos, float2 Rotate);
	void Update(int64 DeltaTIme,
		uint KeyBoardActionBitMask,
		uint MouseActionBitMask,
		double2 MousePos,
		double2 MousePosPrev);
	CameraData ExportCameraData(int width, int height);
};
//实现场景管理
//一个物体包括一个网格体，一个底层加速结构， 一份模型和材质数据，若干sbt记录
//渲染准备部分执行的逻辑有：
//加载模型、生成GAS、TAS
//创建程序组、创建管线
//创建sbt records
//设置根签名
using namespace std;
//要统一每个物体的sbt数量
struct ObjectDesc {
	Mesh mesh;
	Material mat;
	vector<string> shaders;
};
struct Object {
	OptixTraversableHandle GASHandle;
	UniquePtrDevice GASOutputBuffer;

	UniquePtrDevice MaterialBuffer;
	UniquePtrDevice NormalBuffer;
	UniquePtrDevice VertexBuffer;
	UniquePtrDevice TexcoordBuffer;
	UniquePtrDevice GeometryBuffer;
	UniquePtrDevice ModelData;

	vector<UniquePtrDevice> SbtRecordsData;
};
// 编写程序化几何体的类
// 用以支持灯光系统
// scene manager中add object方法负责构建GAS和创建一条SBT Record
// scene manager中build scene方法负责构建TAS和着色器绑定表
// 创建Procedural object类
// 物体类包含模型数据，模型数据包含几何数据和材质数据，还要记录Record的内容
struct ProceduralObject{
	OptixTraversableHandle GASHandle;
	UniquePtrDevice GASOutputBuffer;
	UniquePtrDevice AabbBuffer;
	// 程序化几何体不存在几何数据，只有材质数据需要上传
	UniquePtrDevice MaterialBuffer;
	vector<UniquePtrDevice> SbtRecordsData;
};

struct RayTracingConfig {
	OptixPipelineCompileOptions pipelineCompileOptions;
	uint NumSbtRecords;
	uint MaxSceneTraversalDepth;
	uint MaxRayRecursiveDepth;
};

typedef unordered_map<string, OptixProgramGroup> ShaderManager;
typedef unordered_map<string, OptixModule> Modules;
typedef unordered_map<string, Object> Objects;
typedef unordered_map<string, ProceduralObject> ProceduralObjects;
//封装场景管理类，管理资源，渲染初始化和每帧的调用
class SceneManager {
private:
	uint MaxSceneTraversalDepth;
	uint MaxRayRecursiveDepth;
	uint NumSbtRecords;
	OptixPipelineCompileOptions pipelineCompileOptions;
	OptixPipeline pipeLine= nullptr;
	OptixDeviceContext Context= nullptr;
public:
	uint GetMaxRecursionDepth();
	void SetRayTracingConfig(RayTracingConfig conf);
	void WarmUp();
	~SceneManager();
private:
	/////////////////////////////////////////////
	// module和shader相关的部分
	//用于管理所有shader的结构
	ShaderManager shaderManager;
	Modules modules;
	OptixProgramGroup ShaderRG=nullptr;
	OptixProgramGroup ShaderMiss= nullptr;
public:
	void DestroyShaderManager();
	void DestroyModules();
	void ImportCompiledShader(string path, string ModuleName);
	void AddRayGenerationShader(string func_name, string module_name);
	void AddHitShader(string ShaderName, string module_name, string func_name_ch, string func_name_ah, string func_name_is);
	void AddMissShader(string func_name, string module_name);

	/////////////////////////////////////////////
	//用于渲染的物体的列表
private:
	Objects objects;
	OptixTraversableHandle TASHandle;
	UniquePtrDevice IASOutputBuffer;
	UniquePtrDevice SbtRecordMiss;
	UniquePtrDevice SbtRecordRG;
	UniquePtrDevice SbtRecordHit;
	UniquePtrDevice SbtRecordException;
	OptixShaderBindingTable Sbt = {};
	// 再加上程序化几何体
	ProceduralObjects proceduralObjects;
public:
	OptixTraversableHandle GetTraversableHandle();
	void AddObjects(ObjectDesc desc,string Name);
	void ConfigureMissSbt(MissData Data);
	void ConfigureRGSbt(RayGenData Data);
	void BuildScene();

	void AddProceduralObject(string name, OptixAabb aabb, ProceduralGeometryMaterialBuffer mat,vector<string> shaders,bool isLight);
	void BuildSceneWithProceduralGeometrySupported();
private:
	UniquePtrDevice LaunchParameter;
public:
	void DispatchRays(uchar4* FrameBuffer, CUstream& Stream, CameraData cameraData, uint Width, uint Height, uint Spp);
	void DispatchRays(uchar4* FrameBuffer, CUstream& Stream, LaunchParameters* LParams, uint Width, uint Height);
};