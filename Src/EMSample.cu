#include "EMSample.cuh"
//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
//#include <thrust/transform_reduce.h>
//#include <thrust/functional.h>
//#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>


inline void cudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line)
{
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
            << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
        std::cerr << ss.str();
        throw std::runtime_error(ss.str().c_str());
    }
}
#define CUDA_CHECK( call ) cudaCheck( call, #call, __FILE__, __LINE__ )


__host__ __device__ __forceinline__ float LuminanceFromFloat4(const float4& c)
{
    // Rec.709 / sRGB luminance, assuming linear RGB
    float y = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    return fmaxf(y, 0.0f);
    //return fmaxf(c.x, fmaxf(c.y, c.z));
}

struct LuminanceCompare {
    __host__ __device__
        bool operator()(const float4& a, const float4& b) const {
        float la = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z;
        float lb = 0.2126f * b.x + 0.7152f * b.y + 0.0722f * b.z;
        return la < lb;  // max_element 用 <
    }
};

__global__ void DownSampleAndGetLuminanceKernel(
    float* __restrict__ luminanceBuffer,   // size: 256 * 128
    const float4* __restrict__ rawImage,   // size: Width * Height
    uint Width,
    uint Height)
{
    const uint dx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= DST_W || dy >= DST_H)
        return;

    // 用整数边界定义 box filter 覆盖范围
    // 当前目标像素 [dx, dx+1) 映射回原图上的区间
    const uint sx0 = (uint)(((unsigned long long)dx * Width) / DST_W);
    const uint sx1 = (uint)(((unsigned long long)(dx + 1) * Width) / DST_W);
    const uint sy0 = (uint)(((unsigned long long)dy * Height) / DST_H);
    const uint sy1 = (uint)(((unsigned long long)(dy + 1) * Height) / DST_H);

    float sum = 0.0f;
    uint count = 0;

    for (uint sy = sy0; sy < sy1; ++sy)
    {
        const uint rowOffset = sy * Width;
        for (uint sx = sx0; sx < sx1; ++sx)
        {
            const float4 c = rawImage[rowOffset + sx];
            float lum = LuminanceFromFloat4(c);
            sum += lum;
            ++count;
        }
    }

    const uint outIndex = dy * DST_W + dx;
    luminanceBuffer[outIndex] = sum / (float)count;
}

extern "C" void DownSampleAndGetLuminance(float* luminanceBuffer, float4* rawImage,uint Width,uint Height) {
    float4* rawImageDevice;
    //thrust::device_ptr<float4> dev_ptr(rawImage);

    //auto iter = thrust::max_element(dev_ptr, dev_ptr + Width*Height-1, LuminanceCompare());

    //float4 brightest = *iter;
    CUDA_CHECK(cudaMalloc(&rawImageDevice, sizeof(float4) * Width * Height));
    CUDA_CHECK(cudaMemcpy(rawImageDevice, rawImage, sizeof(float4) * Width * Height, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((DST_W + block.x - 1) / block.x,
        (DST_H + block.y - 1) / block.y);

    DownSampleAndGetLuminanceKernel << <grid, block >> > (
        luminanceBuffer,
        rawImageDevice,
        Width,
        Height
        );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(rawImageDevice));
}
