#include "EMSample.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

__global__ void BuildTables(uint w, uint h, uint* quantified_prefixsum_prob, uint* indextosample, uint* prob) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (idx >= w * h) { return; }
    // 二分查找
    uint Left = 0;
    uint Right = w * h - 1;
    while (Right > Left) {
        uint mid = (Left + Right) / 2;
        if (quantified_prefixsum_prob[mid] > idx) {
            // 命中,向左查找
            Right = mid;
        }
        else {
            Left = mid;
        }
        if (Left + 1 == Right) {
            break;
        }
    }
    uint id;
    if (quantified_prefixsum_prob[Left] > idx) {
        id = Left;
    }
    else if (quantified_prefixsum_prob[Right] > idx) {
        id = Right;
    }
    else {
        printf("没找到，ErrIdx:%d\n", idx);
        assert(false);
    }
    indextosample[idx] = id;
    prob[idx] = quantified_prefixsum_prob[id] - (id > 0 ? quantified_prefixsum_prob[id - 1] : 0);
}

template <typename T>
void print_device_vector(const thrust::device_vector<T>& d_vec) {
    thrust::host_vector<T> h_vec = d_vec; // 复制到 host 端
    std::cout << "[";
    for (size_t i = 0; i < h_vec.size(); i++) {
        std::cout << h_vec[i];
        if (i < h_vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}


 extern "C" void GenerateQuantificationPrefixSumLookupTable(double* data, uint w, uint h, uint*&QuantificationProb,uint*&SampleIndex, bool debug = false)
{
    using namespace thrust;
    // 首先创建设备端的gpu数组
    device_vector<double> image_luminance_device(data, data+w * h);
    double luminance_sum = reduce(image_luminance_device.begin(), image_luminance_device.end(), 0.0, thrust::plus<double>());
    if (debug) {
        std::cout << "亮度";
        print_device_vector(image_luminance_device);
    }
    
    // 均值归一化
    transform(image_luminance_device.begin(), image_luminance_device.end(), image_luminance_device.begin(),
        [luminance_sum, w, h] __device__(double& x) { return w * h * x / luminance_sum; });

    if (debug) {
        std::cout << "概率";
        print_device_vector(image_luminance_device);
    }
    
    // 前缀和
    inclusive_scan(image_luminance_device.begin(), image_luminance_device.end(), image_luminance_device.begin());

    if (debug) {
        std::cout << "概率前缀和";
        print_device_vector(image_luminance_device);
    }
    
    // 量化过程中，若一个点的采样概率过低，就会被忽略，重要性采样就是要采样亮的地方
    device_vector<uint> quantified_prefixsum_prob(w * h);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(image_luminance_device.begin(), thrust::counting_iterator<int>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(image_luminance_device.end(), thrust::counting_iterator<int>(w * h))),
        quantified_prefixsum_prob.begin(),
        [w, h] __device__(thrust::tuple<double, int> t) {
        double x = thrust::get<0>(t);  // 获取数据值
        int index = thrust::get<1>(t); // 获取索引

        // 如果是最后一个索引，返回 index + 1，否则返回 floor(x)
        return (index == w * h - 1) ? (uint)(index + 1) : (uint)floor(x);
    });

    if (debug) {
        std::cout << "量化概率累计";
        print_device_vector(quantified_prefixsum_prob);
    }
    
    device_vector<uint> prob(w * h);
    device_vector<uint> indextosample(w * h);

    BuildTables << <1 + (w * h / 128), 128 >> > (w, h, raw_pointer_cast(quantified_prefixsum_prob.data()), raw_pointer_cast(indextosample.data()), raw_pointer_cast(prob.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (debug) {
        std::cout << "样本";
    print_device_vector(indextosample);
    std::cout << "量化概率";
    print_device_vector(prob);
    }
    

    QuantificationProb = (uint*)malloc(w * h * sizeof(uint));
    SampleIndex = (uint*)malloc(w * h * sizeof(uint));
    CUDA_CHECK(cudaMemcpy(QuantificationProb, raw_pointer_cast(prob.data()), sizeof(uint) * w * h, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(SampleIndex, raw_pointer_cast(indextosample.data()), sizeof(uint) * w * h, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}
 extern "C" void GenerateQuantificationPrefixSumLookupTable_float4(float4* data, uint w, uint h, uint * &QuantificationProb, uint * &SampleIndex, bool debug=false)
{
    using namespace thrust;
    // 首先创建设备端的gpu数组
    device_vector<float4> image_device(data, data + w * h);
    device_vector<double> image_luminance_device(w * h);
    transform(image_device.begin(), image_device.end(), image_luminance_device.begin(),
        []__device__(float4 & color) {
        double luminance = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
        return log(luminance + 1 + 1e-7f);
    });
    double* tmp = (double*)malloc(sizeof(double) * w * h);
    CUDA_CHECK(cudaMemcpy(tmp, raw_pointer_cast(image_luminance_device.data()), sizeof(double)* w* h, cudaMemcpyDeviceToHost));
    GenerateQuantificationPrefixSumLookupTable(tmp, w, h, QuantificationProb, SampleIndex, debug);
    free(tmp);
}