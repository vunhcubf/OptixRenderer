#pragma once
#define IMATH_HALF_NO_LOOKUP_TABLE
#include "common.h"
#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include "cuda.h"
#include "cuda_runtime.h"

static inline float4* ReadOpenExr(string path, uint& width, uint& height) {
	Imf::RgbaInputFile file(path.c_str());
	Imath::Box2i       dw = file.dataWindow();
	int                w = dw.max.x - dw.min.x + 1;
	int                h = dw.max.y - dw.min.y + 1;
	height = h;
	width = w;
	Imf::Array2D<Imf::Rgba> pixels(w, h);

	file.setFrameBuffer(&pixels[0][0], 1, w);
	file.readPixels(dw.min.y, dw.max.y);

	float4* img = (float4*)malloc(sizeof(float4) * w * h);
	for (uint iw = 0; iw < w; iw++) {
		for (uint ih = 0; ih < h; ih++) {
			img[ih + iw * h].x = pixels[w - 1 - iw][ih].r;
			img[ih + iw * h].y = pixels[w - 1 - iw][ih].g;
			img[ih + iw * h].z = pixels[w - 1 - iw][ih].b;
			img[ih + iw * h].w = pixels[w - 1 - iw][ih].a;
		}
		cout << endl;
	}
	return img;
}

template<typename T>
class MyTexture {
private:
	cudaArray_t cuArray = nullptr;
	cudaChannelFormatDesc channelDesc;
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaTextureObject_t texObj = 0;
public:
	cudaTextureObject_t GetTextureId() {
		return texObj;
	}
	~MyTexture() {
		// Destroy texture object
		CUDA_CHECK(cudaDestroyTextureObject(texObj));

		// Free device memory
		if (cuArray) {
			CUDA_CHECK(cudaFreeArray(cuArray));
		}
	}
	MyTexture() = delete;
	MyTexture(T* h_data,uint width, uint height) {
		channelDesc = cudaCreateChannelDesc<T>();
		CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
		const size_t spitch = width * sizeof(T);
		// Copy data located at address h_data in host memory to device memory
		CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(T),height, cudaMemcpyHostToDevice));
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeMirror;
		texDesc.addressMode[1] = cudaAddressModeMirror;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = true;

		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
		if (!cuArray) {
			throw std::exception("空的纹理数组");
		}
		CUDA_CHECK(cudaDeviceSynchronize());
	}
};
template<>
class MyTexture<float4> {
private:
	cudaArray_t cuArray = nullptr;
	cudaChannelFormatDesc channelDesc;
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaTextureObject_t texObj = 0;
public:
	cudaTextureObject_t GetTextureId() {
		return texObj;
	}
	~MyTexture() {
		// Destroy texture object
		CUDA_CHECK(cudaDestroyTextureObject(texObj));

		// Free device memory
		if (cuArray) {
			CUDA_CHECK(cudaFreeArray(cuArray));
		}
	}
	MyTexture() = delete;
	MyTexture(float4* h_data, uint width, uint height) {
		channelDesc = cudaCreateChannelDesc<float4>();
		CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
		const size_t spitch = width * sizeof(float4);
		// Copy data located at address h_data in host memory to device memory
		CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float4), height, cudaMemcpyHostToDevice));
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeMirror;
		texDesc.addressMode[1] = cudaAddressModeMirror;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = true;

		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
		if (!cuArray) {
			throw std::exception("空的纹理数组");
		}
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	MyTexture(string path) {
		uint width, height;
		float4* h_data = ReadOpenExr(path, width, height);
		channelDesc = cudaCreateChannelDesc<float4>();
		CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
		const size_t spitch = width * sizeof(float4);
		// Copy data located at address h_data in host memory to device memory
		CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float4), height, cudaMemcpyHostToDevice));
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeMirror;
		texDesc.addressMode[1] = cudaAddressModeMirror;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = true;

		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
		if (!cuArray) {
			throw std::exception("空的纹理数组");
		}
		CUDA_CHECK(cudaDeviceSynchronize());
		free(h_data);
	}
};
