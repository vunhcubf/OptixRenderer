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
		BitMask |= (1U << type);  // ʹ�� OR ����������λΪ 1
	}
	else {
		BitMask &= ~(1U << type); // ʹ�� AND ����������λΪ 0
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
//ʵ�ֳ�������
//һ���������һ�������壬һ���ײ���ٽṹ�� һ��ģ�ͺͲ������ݣ�����sbt��¼
//��Ⱦ׼������ִ�е��߼��У�
//����ģ�͡�����GAS��TAS
//���������顢��������
//����sbt records
//���ø�ǩ��
using namespace std;
//Ҫͳһÿ�������sbt����
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
// ��д���򻯼��������
// ����֧�ֵƹ�ϵͳ
// scene manager��add object�������𹹽�GAS�ʹ���һ��SBT Record
// scene manager��build scene�������𹹽�TAS����ɫ���󶨱�
// ����Procedural object��
// ���������ģ�����ݣ�ģ�����ݰ����������ݺͲ������ݣ���Ҫ��¼Record������
struct ProceduralObject{
	OptixTraversableHandle GASHandle;
	UniquePtrDevice GASOutputBuffer;
	UniquePtrDevice AabbBuffer;
	// ���򻯼����岻���ڼ������ݣ�ֻ�в���������Ҫ�ϴ�
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
//��װ���������࣬������Դ����Ⱦ��ʼ����ÿ֡�ĵ���
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
	// module��shader��صĲ���
	//���ڹ�������shader�Ľṹ
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
	//������Ⱦ��������б�
private:
	Objects objects;
	OptixTraversableHandle TASHandle;
	UniquePtrDevice IASOutputBuffer;
	UniquePtrDevice SbtRecordMiss;
	UniquePtrDevice SbtRecordRG;
	UniquePtrDevice SbtRecordHit;
	UniquePtrDevice SbtRecordException;
	OptixShaderBindingTable Sbt = {};
	// �ټ��ϳ��򻯼�����
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