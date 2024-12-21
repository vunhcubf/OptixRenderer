#pragma once
#include "common.cuh"
#include "tonemapping.cuh"
#include "bxdf.cuh"
//用于多帧合成的核函数

extern "C" __global__ void AccumulateFrame(uint PixelCount, uint64 FrameCounter, uchar4 * OutputSRGBBuffer, float3 * IndirectOutputBuffer, float3 * AccumulateBuffer) {
	uint Idx=threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= PixelCount) { return; }
//#define DEBUG_RENDER_DISABLE_ACCUMULATION
#ifndef DEBUG_RENDER_DISABLE_ACCUMULATION
	//帧计数器表示前面已渲染了N帧
	float3 AccumulatedColor;
	if (FrameCounter == 0) {
		AccumulatedColor = IndirectOutputBuffer[Idx];
	}
	else {
		AccumulatedColor = FrameCounter * AccumulateBuffer[Idx] + IndirectOutputBuffer[Idx];
	}
	
	AccumulateBuffer[Idx] = AccumulatedColor / (FrameCounter + 1);
	OutputSRGBBuffer[Idx] = make_color(ACESFilm(AccumulatedColor / (FrameCounter + 1)));
#else
	float3 c = IndirectOutputBuffer[Idx];
	if (isnan(c)) {
		c = make_float3(1, 0, 1);
	}
	else if(isinf(c)) {
		c = make_float3(0, 1, 0);
	}
	else if (c.x < 0 || c.y < 0 || c.z < 0) {
		c = make_float3(1, 0, 0);
	}
	OutputSRGBBuffer[Idx] = make_color(c);
#endif
}