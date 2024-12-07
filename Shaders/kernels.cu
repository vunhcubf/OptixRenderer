#pragma once
#include "common.cuh"
#include "tonemapping.cuh"
//用于多帧合成的核函数

extern "C" __global__ void AccumulateFrame(uint PixelCount, uint64 FrameCounter, uchar4 * OutputSRGBBuffer, float3 * IndirectOutputBuffer, float3 * AccumulateBuffer) {
	uint Idx=threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= PixelCount) { return; }
	//帧计数器表示前面已渲染了N帧
	float3 AccumulatedColor = FrameCounter * fmaxf(make_float3(0), AccumulateBuffer[Idx]) + fmaxf(make_float3(0), IndirectOutputBuffer[Idx]);
	// float brightness = length(AccumulatedColor);
	// if (brightness < 0.0f) {
	// 	AccumulatedColor = make_float3(0, 0, 1);
	// }
	// else if (brightness > 1e6f) {
	// 	AccumulatedColor = make_float3(1, 0, 0);
	// }
	AccumulateBuffer[Idx] = AccumulatedColor / (FrameCounter + 1);
	OutputSRGBBuffer[Idx] = make_color(ACESFilm(AccumulatedColor / (FrameCounter + 1)));
}