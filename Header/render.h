#pragma once


#include <optix.h>

#include <optix_stack_size.h>


#include <cuda_runtime.h>

#include "Exception.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <array>

#include "common.h"
#include "mesh.h"

void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */);

OptixTraversableHandle CreateGAS(
	OptixDeviceContext& Context,
	MyMesh& mesh,
	uint SbtNumRecord,
	std::vector<CUdeviceptr>& GpuBufferToRelease);

OptixTraversableHandle CreateIAS(
	OptixDeviceContext& Context,
	OptixInstance* InstListDevice,
	uint NumInsts,
	std::vector<CUdeviceptr>& GpuBufferToRelease);

OptixDeviceContext CreateContext();

//typedef enum OptixTraversableGraphFlags
//{
//	///  Used to signal that any traversable graphs is valid.
//	///  This flag is mutually exclusive with all other flags.
//	OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY = 0,
//
//	///  Used to signal that a traversable graph of a single Geometry Acceleration
//	///  Structure (GAS) without any transforms is valid. This flag may be combined with
//	///  other flags except for OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY.
//	OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1u << 0,
//
//	///  Used to signal that a traversable graph of a single Instance Acceleration
//	///  Structure (IAS) directly connected to Geometry Acceleration Structure (GAS)
//	///  traversables without transform traversables in between is valid.  This flag may
//	///  be combined with other flags except for OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY.
//	OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING = 1u << 1,
//} OptixTraversableGraphFlags;
OptixPipelineCompileOptions CreatePipelineCompileOptions(
	uint TraversableGraphFlags, 
	int NumPayloadValue, 
	int NumAttributeValues);

OptixModule CreateModule(
	std::string shader_path, 
	OptixDeviceContext& Context, 
	OptixPipelineCompileOptions PipelineCompileOpts);

OptixProgramGroup CreateRayGenPg(
	OptixDeviceContext& Context,
	OptixModule& Module,
	std::string EntryFunctionName);

OptixProgramGroup CreateHitGroupPg(
	OptixDeviceContext& Context, 
	OptixModule& Module, 
	std::string EntryFunctionNameCH, 
	std::string EntryFunctionNameAH, 
	std::string EntryFunctionNameIS);

OptixProgramGroup CreateMissPg(
	OptixDeviceContext& Context, 
	OptixModule& Module, 
	std::string EntryFunctionName);

OptixPipeline CreatePipeline(
	OptixDeviceContext& Context, 
	OptixPipelineCompileOptions PipelineCompileOpts, 
	std::vector<OptixProgramGroup> PgUsed,
	uint MaxRayRecursiveDepth,uint MaxSceneTraversalDepth);
OptixPipeline CreatePipeline(
	OptixDeviceContext& Context,
	OptixPipelineCompileOptions PipelineCompileOpts,
	OptixProgramGroup* PgUsed,
	uint NumOfPg,
	uint MaxRayRecursiveDepth, uint MaxSceneTraversalDepth);

template<typename T>
CUdeviceptr CreateSbtRecord(OptixProgramGroup& ProgramGroup, T Data) {
	SbtRecord<T>* RecordDevice;
	SbtRecord<T> RecordHost;
	//头部信息
	OPTIX_CHECK(optixSbtRecordPackHeader(ProgramGroup, &RecordHost));
	//具体信息
	RecordHost.data = Data;
	CUDA_CHECK(cudaMalloc((void**) & RecordDevice, sizeof(SbtRecord<T>)));
	CUDA_CHECK(cudaMemcpy((void*)RecordDevice, &RecordHost, sizeof(SbtRecord<T>), cudaMemcpyHostToDevice));
	return (CUdeviceptr)RecordDevice;
}

ModelData* AssembleModelData(Material mat, MyMesh& mesh, std::vector<CUdeviceptr>& GpuBuffersToRelease);

void ReleaseGpuResources(std::vector<CUdeviceptr>& GpuBufferToRelease);
CUdeviceptr UploadAnything(HostMemStream mem, std::vector<CUdeviceptr>& GpuBufferToRelease);
void* UploadAnything(void* ptr,size_t size);
GeometryBuffer* CreateAndUploadGeometryBuffer(MyMesh& mesh, std::vector<CUdeviceptr>& GpuBuffersToRelease);