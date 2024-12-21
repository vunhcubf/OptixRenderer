#include "common.h"

__device__ __host__ float3 float3_add(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ float3 float3_minus(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ float3 float3_multiply(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__ float3 float3_divide(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __host__ float3 float3_scale(const float3& a, const float& b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ __host__ float3 CrossProduct(float3 a, float3 b)
{
    float x = a.y * b.z - a.z * b.y;
    float y = a.z * b.x - a.x * b.z;
    float z = a.x * b.y - a.y * b.x;
    return make_float3(x, y, z);
}

std::string ReadOptixir(std::string& path) {
    std::ifstream file(path.c_str(), std::ios::binary);
    std::string code;
    if (file.good())
    {
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        code.assign(buffer.begin(), buffer.end());
        return code;
    }
    throw std::runtime_error("invalid optixir file");
    return "";
}

void* GeometryBuffer::GetNormalPtr()
{
    return (void*)Normal;
}

void* GeometryBuffer::GetUvPtr()
{
    return (void*)uv;
}

void* GeometryBuffer::GetVerticesPtr()
{
    return (void*)Vertices;
}
void** GeometryBuffer::GetNormalPPtr()
{
    return (void**)(&Normal);
}

void** GeometryBuffer::GetUvPPtr()
{
    return (void**)(&uv);
}

void** GeometryBuffer::GetVerticesPPtr()
{
    return (void**)(&Vertices);
}

HostMemStream HostMemStream::GetEmpty()
{
    return {nullptr,0};
}

HostMemStream HostMemStream::FromCudaDevicePtr(CUdeviceptr ptr)
{
    HostMemStream a={(void*)ptr,sizeof(CUdeviceptr)};
    return a;
}
