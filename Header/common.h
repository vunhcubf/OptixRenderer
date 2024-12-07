#pragma once
#include <windows.h>
#include <vector_types.h>
#include <tuple>
#include <string>
#include <vector>
#include <fstream>
#include "Exception.h"
#include <nvrtc.h>
#include <vector_functions.h>
#include <optix.h>
#include <iostream>
#include <unordered_set>
#include <exception>
#include <functional>
#include <unordered_map>
#include <sstream>
#include <curand.h>
#include <curand_kernel.h>
#include <filesystem>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC 
#include "stb/include/stb_image.h"
#include <assert.h>
typedef unsigned int uint;
typedef unsigned long long uint64;
typedef long long int64;
typedef float float32;
typedef double float64;

// 前向声明
struct TextureView;
using std::string;
using namespace::std;
#define NO_TEXTURE_HERE 0xFFFFFFFF
template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
#define CURAND_CHECK(x) assert(x==CURAND_STATUS_SUCCESS)
#define M_PI 3.14159265358979f
#define M_REVERSE_PI 0.318309886183791f
struct HostMemStream {
    void* Ptr;
    size_t SizeInBytes;
    static HostMemStream GetEmpty();
    static HostMemStream FromCudaDevicePtr(CUdeviceptr ptr);
};

enum MaterialType {
    MATERIAL_AREALIGHT,
    MATERIAL_OBJ
};

struct AreaLight {
    float3 P1, P2, P3, P4, Color;//p1,p2,p3,p4从下面看顺时针一圈
    float Area;
};

struct CameraData {
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
};

struct LaunchParametersDesc {
    CameraData cameraData;
    AreaLight areaLight;
};

struct GeometryBuffer {
    CUdeviceptr Normal=(CUdeviceptr)nullptr;
    CUdeviceptr Vertices = (CUdeviceptr)nullptr;
    CUdeviceptr uv = (CUdeviceptr)nullptr;
    void* GetNormalPtr();
    void* GetUvPtr();
    void* GetVerticesPtr();
    void** GetNormalPPtr();
    void** GetUvPPtr();
    void** GetVerticesPPtr();
};

struct RayGenData
{
    float r, g, b;
    cudaTextureObject_t TestTex;
};
struct MissData {
    float3 BackgroundColor;
    float SkyBoxIntensity;
    cudaTextureObject_t SkyBox;
};

struct SbtDataStruct
{
    CUdeviceptr DataPtr;
};
//原理化BSDF
struct Material
{
    cudaTextureObject_t NormalMap=NO_TEXTURE_HERE;
    cudaTextureObject_t BaseColorMap= NO_TEXTURE_HERE;
    cudaTextureObject_t ARMMap= NO_TEXTURE_HERE;
    float3 BaseColor = make_float3(0.8,0.8,0.8);
    float3 Emission = make_float3(0, 0, 0);
    float Roughness=0.5f;
    float Metallic = 0.0f;
    float Specular=1.f;
    float Transmission = 0.0f;
    float Ior = 1.4f;
    float SpecularTint=0.0f;
    float Opacity = 1.0f;
    MaterialType MaterialType = MaterialType::MATERIAL_OBJ;
};

inline void ResetMaterial(Material& mat) {
    mat.NormalMap = NO_TEXTURE_HERE;
    mat.BaseColorMap = NO_TEXTURE_HERE;
    mat.ARMMap = NO_TEXTURE_HERE;
    mat.BaseColor = make_float3(0.4, 0.4, 0.4);
    mat.Emission = make_float3(0, 0, 0);
    mat.Roughness = 0.5f;
    mat.Metallic = 0.0f;
    mat.Specular = 1.f;
    mat.Transmission = 0.0f;
    mat.Ior = 1.4f;
    mat.SpecularTint = 0.0f;
    mat.Opacity = 1.0f;
    mat.MaterialType = MaterialType::MATERIAL_OBJ;
}
struct BlueNoiseMapBuffer{
    unsigned char* Data;
    int width;
    int height;
    int channel;
};
struct BlueNoiseMapBufferManager{
public:
    unsigned char* hostdata;
    unsigned char* devicedata;
    BlueNoiseMapBuffer* devicebuffer;
    int width;
    int height;
    int channel;
public:
    inline BlueNoiseMapBuffer* GetBuffer(){
        return devicebuffer;
    }
    inline BlueNoiseMapBufferManager(const char* path){
        this->hostdata = stbi_load(path, &this->width, &this->height, &this->channel, 0);
        // 自动上传数据到gpu
        CUDA_CHECK(cudaMalloc(&devicedata,sizeof(unsigned char)*width*height*channel));
        CUDA_CHECK(cudaMemcpy(devicedata,hostdata,sizeof(unsigned char)*width*height*channel,cudaMemcpyHostToDevice));
        BlueNoiseMapBuffer hostbuffer;
        hostbuffer.width=this->width;
        hostbuffer.height=this->height;
        hostbuffer.channel=this->channel;
        hostbuffer.Data=this->devicedata;
        CUDA_CHECK(cudaMalloc(&devicebuffer,sizeof(BlueNoiseMapBuffer)));
        CUDA_CHECK(cudaMemcpy(devicebuffer,&hostbuffer,sizeof(BlueNoiseMapBuffer),cudaMemcpyHostToDevice));
    }
    inline ~BlueNoiseMapBufferManager(){
        CUDA_CHECK(cudaFree(devicebuffer));
        CUDA_CHECK(cudaFree(devicedata));
        stbi_image_free(hostdata);
    }
};
struct LaunchParameters {
    float3* IndirectOutputBuffer;
    uchar4* ImagePtr;
    uint Width;
    uint Height;
    CameraData cameraData;
    OptixTraversableHandle Handle;
    AreaLight areaLight;
    uint Seed;
    uint64 FrameNumber;
    uint Spp;
    uint MaxRecursionDepth;
    // 随机数生成
    uint64* PixelOffset;
	BlueNoiseMapBuffer* BlueNoiseBuffer;
	TextureView* textureDiscriptorsRange;
};
struct ModelData {
    GeometryBuffer* GeometryData;
    Material* MaterialData;
};
enum
{
    DEVICE_PTR,
    HOST_PTR_NEW,
    HOST_PTR_MALLOC
};

//所有的数组都用malloc
//所有类都用new
template<typename T, int type>
class UniquePtr {
private:
    T* Ptr=nullptr;
public:
    //无参构造
    UniquePtr() { Ptr = nullptr; }
    //拷贝构造
    UniquePtr(const UniquePtr&) = delete;
    //移动构造
    UniquePtr(const UniquePtr&& in) { this->Ptr = in.Ptr; }
    ~UniquePtr() { Destroy(); }
    UniquePtr& operator=(T*& p) {
        this->Ptr = p;
        p = nullptr;
        return *this;
    }
    UniquePtr& operator=(T*&& p) {
        this->Ptr = p;
        p = nullptr;
        return *this;
    }
    UniquePtr& operator=(UniquePtr&& other) {
        if (this != &other) {
            Destroy();
            this->Ptr = other.Ptr;
            other.Ptr = nullptr;
        }
        return *this;
    }
    UniquePtr& operator=(UniquePtr& other) {
        if (this != &other) {
            Destroy();
            this->Ptr = other.Ptr;
            other.Ptr = nullptr;
        }
        return *this;
    }
    T* GetPtr() {
        if(Ptr){
            return Ptr;
        }
        else {
            std::cout << "Attempt to obtain a non-owning unique pointer" << std::endl;
            return nullptr;
        }
    }
    T** GetAddressOfPtr() { return &Ptr; }
    void Destroy() {
        if (this != nullptr && Ptr != nullptr) {
            if constexpr (type == DEVICE_PTR) {
                cudaError_t error = cudaFree(Ptr);
                if (error != cudaSuccess)
                {
                    std::stringstream ss;
                    ss << "CUDA call (" << cudaGetErrorString(error) << " ) failed with error: '"
                        << cudaGetErrorString(error) << "' (" << __FILE__ << ":" << __LINE__ << ")\n";
                    std::cerr << ss.str();
                    //throw std::exception(ss.str().c_str());
                }
            }
            else if constexpr (type == HOST_PTR_NEW) {
                delete Ptr;
            }
            else if constexpr (type == HOST_PTR_MALLOC) {
                free(Ptr);
            }
            Ptr = nullptr;
        }
    }
};
typedef UniquePtr<void, DEVICE_PTR> UniquePtrDevice;
std::string ReadOptixir(std::string& path);

__device__ __host__ float3 float3_add(const float3& a, const float3& b);
__device__ __host__ float3 float3_minus(const float3& a, const float3& b);
__device__ __host__ float3 float3_multiply(const float3& a, const float3& b);
__device__ __host__ float3 float3_divide(const float3& a, const float3& b);
__device__ __host__ float3 float3_scale(const float3& a, const float& b);

__device__ __host__ float3 CrossProduct(float3 a, float3 b);

inline void writeToFile(const std::string& filename, const std::string& content, bool append = false) {
    std::ofstream outFile;
    if (append) {
        outFile.open(filename, std::ios::app);
    } else {
        outFile.open(filename);
    }
    if (outFile.is_open()) {
        outFile << content << std::endl;
        outFile.close();
    } else {
        std::cerr << "Failed to open the file: " << filename << std::endl;
    }
}
inline std::string getExecutablePath() {
    char path[MAX_PATH];
    if (GetModuleFileNameA(NULL, path, MAX_PATH) > 0) {
        return std::string(path);
    }
    return "";
}
inline std::string getParentDir(std::string path) {
    std::filesystem::path filepath=path;
    return filepath.parent_path().string();
}
const uint default_width = 1024;
const uint default_height = 1024;