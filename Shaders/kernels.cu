#pragma once
#include "common.cuh"
#include "tonemapping.cuh"
#include "bxdf.cuh"
//���ڶ�֡�ϳɵĺ˺���

extern "C" __global__ void AccumulateFrame(uint PixelCount, uint64 FrameCounter, uchar4 * OutputSRGBBuffer, float3 * IndirectOutputBuffer, float3 * AccumulateBuffer) {
	uint Idx=threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= PixelCount) { return; }
	//֡��������ʾǰ������Ⱦ��N֡
	float3 AccumulatedColor;
	if (FrameCounter == 0) {
		AccumulatedColor = IndirectOutputBuffer[Idx];
	}
	else {
		AccumulatedColor = FrameCounter * AccumulateBuffer[Idx] + IndirectOutputBuffer[Idx];
	}
	
	AccumulateBuffer[Idx] = AccumulatedColor / (FrameCounter + 1);
	OutputSRGBBuffer[Idx] = make_color(ACESFilm(AccumulatedColor / (FrameCounter + 1)));
}