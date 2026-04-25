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
#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda_runtime.h>

inline void SaveUchar4Image(uchar4* data, int width, int height, const char* filename)
{
    // 1. 创建 Mat（4通道）
    cv::Mat img(height, width, CV_8UC4, data);

    // 2. RGBA → BGRA
    cv::Mat img_bgra;
    cv::cvtColor(img, img_bgra, cv::COLOR_RGBA2BGRA);

    // 3. 水平翻转（左右翻转）
    cv::Mat flipped;
    cv::flip(img_bgra, flipped, 1);  // 1 = 水平翻转

    // 4. 保存
    cv::imwrite(filename, flipped);
}

inline std::string GetTimeString()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);  // Windows
#else
    localtime_r(&t, &tm);  // Linux / Mac
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    // 例如：20260415_143025

    return oss.str();
}

template<typename T>
void ShowImage(
    const T* data,
    int width,
    int height,
    bool isDevice,
    const char* windowName = "",
    bool normalize = false,
    bool colormap = false,
    bool logTransform = false,
    bool enableUpscale = true)
{
    if (data == nullptr)
    {
        printf("ShowImage: data is null\n");
        return;
    }

    // -----------------------------
    // 1. 准备 host 数据
    // -----------------------------
    std::vector<T> h_data(width * height);

    if (isDevice)
    {
        CUDA_CHECK(cudaMemcpy(h_data.data(), data,
            sizeof(T) * width * height,
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else
    {
        memcpy(h_data.data(), data, sizeof(T) * width * height);
    }

    // -----------------------------
    // 2. 分类型处理
    // -----------------------------
    cv::Mat display;

    if constexpr (std::is_same_v<T, float>)
    {
        // ---------- float (单通道) ----------
        cv::Mat img(height, width, CV_32F, h_data.data());

        if (normalize)
            cv::normalize(img, display, 0, 255, cv::NORM_MINMAX);
        else
            display = img * 255.0f;

        display.convertTo(display, CV_8U);

        if (colormap)
        {
            cv::Mat colored;
            cv::applyColorMap(display, colored, cv::COLORMAP_JET);
            display = colored;
        }
    }
    else if constexpr (std::is_same_v<T, float4>)
    {
        // ---------- float4 (RGB) ----------
        cv::Mat img(height, width, CV_32FC3);

        for (int i = 0; i < width * height; i++)
        {
            float r = h_data[i].x;
            float g = h_data[i].y;
            float b = h_data[i].z;

            if (logTransform)
            {
                r = logf(fmaxf(r, 0.0f) + 1.0f);
                g = logf(fmaxf(g, 0.0f) + 1.0f);
                b = logf(fmaxf(b, 0.0f) + 1.0f);
            }

            img.at<cv::Vec3f>(i / width, i % width) =
                cv::Vec3f(b, g, r);
        }

        if (normalize)
            cv::normalize(img, display, 0, 255, cv::NORM_MINMAX);
        else
            display = img * 255.0f;

        display.convertTo(display, CV_8UC3);
    }
    else
    {
        printf("ShowImageT: unsupported type\n");
        return;
    }

    // -----------------------------
    // 3. 放大逻辑（带 4K 限制）
    // -----------------------------
    int scale = 1;

    if (enableUpscale)
    {
        const int targetPixels = 1024 * 1024;
        const int maxDisplayW = 3840;
        const int maxDisplayH = 2160;

        int currentPixels = width * height;

        int minScale = 1;
        if (currentPixels < targetPixels)
        {
            float ratio = (float)targetPixels / (float)currentPixels;
            minScale = (int)std::ceil(std::sqrt(ratio));
        }

        int maxScaleW = maxDisplayW / width;
        int maxScaleH = maxDisplayH / height;
        int maxScale = std::min(maxScaleW, maxScaleH);

        if (maxScale < 1) maxScale = 1;

        scale = std::min(minScale, maxScale);
    }

    int newW = width * scale;
    int newH = height * scale;

    cv::Mat finalImage;

    if (scale > 1)
    {
        cv::resize(display, finalImage,
            cv::Size(newW, newH),
            0, 0,
            cv::INTER_NEAREST);
    }
    else
    {
        finalImage = display;
    }

    // -----------------------------
    // 4. 标题
    // -----------------------------
    char title[256];
    snprintf(title, sizeof(title),
        "%s | %dx%d -> %dx%d | scale=%dx",
        windowName, width, height, newW, newH, scale);

    // -----------------------------
    // 5. 显示
    // -----------------------------
    cv::imshow(title, finalImage);
    cv::waitKey(0);
}