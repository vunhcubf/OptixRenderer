#pragma once
#include "common.cuh"
#include "tonemapping.cuh"
#include "bxdf.cuh"
//用于多帧合成的核函数

extern "C" GLOBAL void AccumulateFrame(uint PixelCount, uint64 FrameCounter, uchar4 * OutputSRGBBuffer, float3 * IndirectOutputBuffer, float3 * AccumulateBuffer,ConsoleOptions* options) {
	uint Idx=threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= PixelCount) { return; }
	//帧计数器表示前面已渲染了N帧
	float3 AccumulatedColor;
	if (FrameCounter == 0 || options->frameAccumulationOptions==FrameAccumulationOptions::ForceOff) {
		AccumulatedColor = IndirectOutputBuffer[Idx];
		OutputSRGBBuffer[Idx] = make_color((ACES2LinearSRGB(ACESFilm(AccumulatedColor))));
		AccumulateBuffer[Idx] = AccumulatedColor;
	}
	else {
		AccumulatedColor = FrameCounter * AccumulateBuffer[Idx] + IndirectOutputBuffer[Idx];
		AccumulateBuffer[Idx] = AccumulatedColor / (FrameCounter + 1);
		OutputSRGBBuffer[Idx] = make_color((ACES2LinearSRGB(ACESFilm(AccumulatedColor / (FrameCounter + 1)))));
	}
}