#pragma once
#define IMATH_HALF_NO_LOOKUP_TABLE
#include "common.h"
#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <memory>

#define TEXTURE_FORMAT_UCHAR1 0
#define TEXTURE_FORMAT_UCHAR2 1
#define TEXTURE_FORMAT_UCHAR3 2
#define TEXTURE_FORMAT_UCHAR4 3
#define TEXTURE_FORMAT_FLOAT1 4
#define TEXTURE_FORMAT_FLOAT2 5
#define TEXTURE_FORMAT_FLOAT3 6
#define TEXTURE_FORMAT_FLOAT4 7

static inline float4* ReadOpenExr(const char* path, uint& width, uint& height) {
	Imf::RgbaInputFile file(path);
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
	}
	return img;
}

static inline unsigned char* ReadLDRImage(const char* path,uint& width,uint& height,uint& channel){
	int w,h,c;
	unsigned char* hostdata = stbi_load(path, &w, &h, &c, 0);
	width=w;
	height=h;
	channel=c;
	return hostdata;
} 

class Texture2D {
private:
	unsigned char textureFormat;
	cudaArray_t cuArray = nullptr;
	cudaChannelFormatDesc channelDesc;
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaTextureObject_t texObj = 0;
	TextureView textureView;
public:
	cudaTextureObject_t GetTextureId() {
		return texObj;
	}
	static inline Texture2D LoadHDRImage(const char* path){
		uint w,h;
		float4* data=ReadOpenExr(path,w,h);
		Texture2D instance=Texture2D(data,w,h,TEXTURE_FORMAT_FLOAT4);
		free(data);
		return instance;
	}
	inline TextureView GetTextureView(){
		return textureView;
	}
	static inline Texture2D LoadLDRImage(const char* path){
		uint w,h,channel;
		unsigned char* data=ReadLDRImage(path,w,h,channel);
		unsigned char textureFormat;
		if(channel==1){
			textureFormat=TEXTURE_FORMAT_UCHAR1;
		}
		else if(channel==2){
			textureFormat=TEXTURE_FORMAT_UCHAR2;
		}
		else if(channel==3){
			textureFormat=TEXTURE_FORMAT_UCHAR3;
		}
		else if(channel==4){
			textureFormat=TEXTURE_FORMAT_UCHAR4;
		}
		else{
			throw std::runtime_error("unsupport channel count when LoadLDRImage");
		}
		Texture2D instance=Texture2D(data,w,h,textureFormat);
		free(data);
		return instance;
	}
	static inline Texture2D LoadImageFromFile(string path){
		 size_t dotPosition = path.find_last_of('.');
		if (dotPosition == std::string::npos) {
			throw std::runtime_error("unsupport no extension file");
		}
		string extension=path.substr(dotPosition + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
		if(extension=="exr"){
			return LoadHDRImage(path.c_str());
		}
		else{
			return LoadLDRImage(path.c_str());
		}
	}
	static inline Texture2D LoadImageFromFile(const char* path){
		return LoadImageFromFile(path);
	}
	~Texture2D() {
		ReleaseGPUResource();
	}
	void ReleaseGPUResource(){
		if (cuArray) {
			CUDA_CHECK(cudaDestroyTextureObject(texObj));
			CUDA_CHECK(cudaFreeArray(cuArray));
			std::cout<<"删除纹理"<<"w: "<<this->textureView.width<<"h: "<<this->textureView.height<<endl;
			cuArray=nullptr;
		}
	}
	Texture2D()=default;
	// 重载拷贝构造和移动构造，以及左值和右值的赋值
	Texture2D(Texture2D& in){
		if(this->cuArray!=in.cuArray){
			ReleaseGPUResource();
			this->textureFormat=in.textureFormat;
			this->cuArray=in.cuArray;
			this->channelDesc=in.channelDesc;
			this->texDesc=in.texDesc;
			this->resDesc=in.resDesc;
			this->texObj=in.texObj;
			this->textureView=in.textureView;
			in.cuArray=nullptr;
		}
	}
	Texture2D(Texture2D&& in){
		if(this->cuArray!=in.cuArray){
			ReleaseGPUResource();
			this->textureFormat=in.textureFormat;
			this->cuArray=in.cuArray;
			this->channelDesc=in.channelDesc;
			this->texDesc=in.texDesc;
			this->resDesc=in.resDesc;
			this->texObj=in.texObj;
			this->textureView=in.textureView;
			in.cuArray=nullptr;
		}
	}
	Texture2D& operator=(Texture2D& in){
		if(this->cuArray!=in.cuArray){
			ReleaseGPUResource();
			this->textureFormat=in.textureFormat;
			this->cuArray=in.cuArray;
			this->channelDesc=in.channelDesc;
			this->texDesc=in.texDesc;
			this->resDesc=in.resDesc;
			this->texObj=in.texObj;
			this->textureView=in.textureView;
			in.cuArray=nullptr;
		}
		return *this;
	}
	Texture2D& operator=(Texture2D&& in){
		if(this->cuArray!=in.cuArray){
			ReleaseGPUResource();
			this->textureFormat=in.textureFormat;
			this->cuArray=in.cuArray;
			this->channelDesc=in.channelDesc;
			this->texDesc=in.texDesc;
			this->resDesc=in.resDesc;
			this->texObj=in.texObj;
			this->textureView=in.textureView;
			in.cuArray=nullptr;
		}
		return *this;
	}
	Texture2D(void* h_data,uint width, uint height,unsigned char textureFormat) {
		size_t sizeOfPixel;
		if(textureFormat==TEXTURE_FORMAT_UCHAR1){
			channelDesc = cudaCreateChannelDesc<uchar1>();
			sizeOfPixel=sizeof(uchar1);
		}
		else if(textureFormat==TEXTURE_FORMAT_UCHAR2){
			channelDesc = cudaCreateChannelDesc<uchar2>();
			sizeOfPixel=sizeof(uchar2);
		}
		else if(textureFormat==TEXTURE_FORMAT_UCHAR3){
			channelDesc = cudaCreateChannelDesc<uchar3>();
			sizeOfPixel=sizeof(uchar3);
		}
		else if(textureFormat==TEXTURE_FORMAT_UCHAR4){
			channelDesc = cudaCreateChannelDesc<uchar4>();
			sizeOfPixel=sizeof(uchar4);
		}
		else if(textureFormat==TEXTURE_FORMAT_FLOAT1){
			channelDesc = cudaCreateChannelDesc<float>();
			sizeOfPixel=sizeof(float);
		}
		else if(textureFormat==TEXTURE_FORMAT_FLOAT2){
			channelDesc = cudaCreateChannelDesc<float2>();
			sizeOfPixel=sizeof(float2);
		}
		else if(textureFormat==TEXTURE_FORMAT_FLOAT3){
			channelDesc = cudaCreateChannelDesc<float3>();
			sizeOfPixel=sizeof(float3);
		}
		else if(textureFormat==TEXTURE_FORMAT_FLOAT4){
			channelDesc = cudaCreateChannelDesc<float4>();
			sizeOfPixel=sizeof(float4);
		}
		else{
			stringstream ss;
			ss << "unsupport texture format! it's " << textureFormat;
			std::cerr << ss.str()<<endl;
			throw std::runtime_error(ss.str());
		}
		CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
		const size_t spitch = width * sizeOfPixel;
		CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeOfPixel,height, cudaMemcpyHostToDevice));
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeMirror;
		texDesc.addressMode[1] = cudaAddressModeMirror;
		texDesc.filterMode = textureFormat<=3? cudaFilterModePoint :cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = true;

		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
		if (!cuArray) {
			throw std::runtime_error("empty texture array!");
		}
		CUDA_CHECK(cudaDeviceSynchronize());
		textureView={width,height,textureFormat,texObj};
	}
};

class TextureManager{
	// 单例类，延长纹理的声明周期
public:
	static inline TextureManager& GetInstance(){
		static TextureManager ins;
		return ins;
	}
	void Add(string name,Texture2D tex){
		if(textures.find(name)==textures.end()){
			textures[name]=tex;
		}
		else{
			throw std::runtime_error("Texture Manager: attempt to add two textures with same name");
		}
	}
private:
	unordered_map<string,Texture2D> textures;
	TextureManager()=default;
	~TextureManager(){
		std::cout<<"正在析构 TextureManager"<<endl;
	}
	TextureManager(const TextureManager&)=delete;
	TextureManager& operator=(const TextureManager&)=delete; 
};