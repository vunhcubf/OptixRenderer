#pragma once

#include "common.h"
#include <string>
#include <sstream>

inline CUmodule LoadModule(std::string path) {
	CUmodule mod;
    CUresult res = cuModuleLoad(&mod, path.c_str());
    if (res != CUDA_SUCCESS) {
        std::stringstream ss;
        ss << "读取文件出错:" << path;
        throw std::runtime_error(ss.str());
    }
    return mod;
}
inline CUfunction LoadFunction(CUmodule module, const char* name) {
    CUfunction func;
    CUresult res = cuModuleGetFunction(&func, module, name);
    if (res != CUDA_SUCCESS) {
        std::stringstream ss;
        ss << "加载函数出错:" << name;
        throw std::runtime_error(ss.str());
    }
    return func;
}

template<typename T>
void* UploadToDevice(T t) {
    void* d_ptr;
    cuMemAlloc((CUdeviceptr*)&d_ptr, sizeof(T));
    cuMemcpy((CUdeviceptr)d_ptr, (CUdeviceptr)&t, sizeof(T));
    return d_ptr;
}