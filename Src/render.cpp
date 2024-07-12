#pragma once
#include "render.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}
OptixTraversableHandle CreateGAS(OptixDeviceContext& Context, MyMesh& mesh, uint SbtNumRecord, std::vector<CUdeviceptr>& GpuBufferToRelease)
{
	OptixTraversableHandle Handle;
	CUdeviceptr GASOutputBuffer;
	OptixAccelBuildOptions AccelBuildOpts = {};
	AccelBuildOpts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;//构建选项，先设置为无，不压缩
	AccelBuildOpts.operation = OPTIX_BUILD_OPERATION_BUILD;
	//上传顶点缓冲
	CUdeviceptr VertexBuffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&VertexBuffer), mesh.GetVerticesCount() * sizeof(float3)));
	CUDA_CHECK(cudaMemcpy((void*)VertexBuffer, mesh.GetVerticesPtr(), mesh.GetVerticesCount() * sizeof(float3), cudaMemcpyHostToDevice));
	//构建设置
	const uint32_t TriangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
	OptixBuildInput TriangleInput = {};
	TriangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	TriangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	TriangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	TriangleInput.triangleArray.numVertices = mesh.GetVerticesCount();
	TriangleInput.triangleArray.vertexBuffers = &VertexBuffer;

	TriangleInput.triangleArray.flags = TriangleInputFlags;
	TriangleInput.triangleArray.numSbtRecords = SbtNumRecord;
	//加速结构大小
	OptixAccelBufferSizes GASBufferSize;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		Context,
		&AccelBuildOpts,
		&TriangleInput,
		1, // Number of build inputs
		&GASBufferSize
	));
	//需要ScratchBuffer
	CUdeviceptr ScratchBuffer;
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&ScratchBuffer),
		GASBufferSize.tempSizeInBytes
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&GASOutputBuffer),
		GASBufferSize.outputSizeInBytes
	));
	OPTIX_CHECK(optixAccelBuild(
		Context,
		0,                  // CUDA stream
		&AccelBuildOpts,
		&TriangleInput,
		1,                  // num build inputs
		ScratchBuffer,
		GASBufferSize.tempSizeInBytes,
		GASOutputBuffer,
		GASBufferSize.outputSizeInBytes,
		&Handle,
		nullptr,            // emitted property list
		0                   // num emitted properties
	));
	GpuBufferToRelease.push_back(GASOutputBuffer);
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ScratchBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(VertexBuffer)));
	return Handle;
}
OptixTraversableHandle CreateIAS(OptixDeviceContext& Context, OptixInstance* InstListDevice, uint NumInsts, std::vector<CUdeviceptr>& GpuBufferToRelease)
{
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixBuildInput InstanceInput = {};
	InstanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	InstanceInput.instanceArray.instances = (CUdeviceptr)InstListDevice;
	InstanceInput.instanceArray.numInstances = NumInsts;

	CUdeviceptr IASOutputBuffer;
	OptixAccelBufferSizes IASBufferSize;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		Context,
		&accelOptions,
		&InstanceInput,
		1, // Number of build inputs
		&IASBufferSize
	));
	//需要ScratchBuffer
	CUdeviceptr ScratchBuffer;
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&ScratchBuffer),
		IASBufferSize.tempSizeInBytes
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&IASOutputBuffer),
		IASBufferSize.outputSizeInBytes
	));
	OptixTraversableHandle InstHandle;
	OPTIX_CHECK(optixAccelBuild(
		Context,
		0,                  // CUDA stream
		&accelOptions,
		&InstanceInput,
		1,                  // num build inputs
		ScratchBuffer,
		IASBufferSize.tempSizeInBytes,
		IASOutputBuffer,
		IASBufferSize.outputSizeInBytes,
		&InstHandle,
		nullptr,            // emitted property list
		0                   // num emitted properties
	));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ScratchBuffer)));
	GpuBufferToRelease.push_back(IASOutputBuffer);
	return InstHandle;
}
OptixDeviceContext CreateContext()
{
	OptixDeviceContext Context = nullptr;
	CUDA_CHECK(cudaFree(0));
	OPTIX_CHECK(optixInit());
	CUcontext CudaContext = 0;
	//创建optix上下文
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
#if defined _DEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
	OPTIX_CHECK(optixDeviceContextCreate(CudaContext, 0, &Context));
	return Context;
}

OptixPipelineCompileOptions CreatePipelineCompileOptions(uint TraversableGraphFlags, int NumPayloadValue, int NumAttributeValues)
{
	OptixPipelineCompileOptions PipelineCompileOpts = {};
	PipelineCompileOpts.usesMotionBlur = false;
	//直接遍历GAS，或者允许套娃一层IAS的GAS
	PipelineCompileOpts.traversableGraphFlags = TraversableGraphFlags;
	PipelineCompileOpts.numPayloadValues = NumPayloadValue;//payload占用3个32位寄存器
	PipelineCompileOpts.numAttributeValues = NumAttributeValues;//相交时的属性数
	//检测所有的异常
#if defined _DEBUG
	PipelineCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_USER;
#endif
	PipelineCompileOpts.pipelineLaunchParamsVariableName = "params";//启动参数的名字，通常是输出的图片
	return PipelineCompileOpts;
}

OptixModule CreateModule(std::string shader_path, OptixDeviceContext& Context, OptixPipelineCompileOptions PipelineCompileOpts)
{
	//创建module，用于封装编译后的着色器，多个模块构成程序组
	OptixModule Module = nullptr;
	//两个编译选项
	//两个编译选项

	OptixModuleCompileOptions ModuleCompileOpts = {};
	ZeroMemory(&ModuleCompileOpts, sizeof(OptixModuleCompileOptions));
#if 0
	//调试模式下不优化，开启调试
	ModuleCompileOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	ModuleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
	ModuleCompileOpts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
	ModuleCompileOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

	//设置好了，该加载shader了
	std::string compiled_shader = ReadOptixir(shader_path);
	OPTIX_CHECK_LOG(optixModuleCreate(
		Context,
		&ModuleCompileOpts,
		&PipelineCompileOpts,
		compiled_shader.c_str(),
		compiled_shader.size(),
		LOG, &LOG_SIZE,
		&Module
	));
	return Module;
}

OptixProgramGroup CreateRayGenPg(OptixDeviceContext& Context, OptixModule& Module, std::string EntryFunctionName)
{
	OptixProgramGroup RayGenGroup = nullptr;
	OptixProgramGroupOptions GroupOpts = {};//不使用payloadtype
	OptixProgramGroupDesc RayGenGroupDesc = {};
	RayGenGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	RayGenGroupDesc.raygen.module = Module;
	RayGenGroupDesc.raygen.entryFunctionName = EntryFunctionName.c_str();
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		Context,
		&RayGenGroupDesc,
		1,   // num program groups
		&GroupOpts,
		LOG, &LOG_SIZE,
		&RayGenGroup
	));
	return RayGenGroup;
}

OptixProgramGroup CreateHitGroupPg(OptixDeviceContext& Context, OptixModule& Module, std::string EntryFunctionNameCH, std::string EntryFunctionNameAH, std::string EntryFunctionNameIS)
{
	OptixProgramGroup HitGroupGroup = nullptr;
	OptixProgramGroupOptions GroupOpts = {};//不使用payloadtype
	OptixProgramGroupDesc HitGroupGroupDesc = {};
	HitGroupGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	if (!EntryFunctionNameCH.empty()) {
		HitGroupGroupDesc.hitgroup.moduleCH = Module;
		HitGroupGroupDesc.hitgroup.entryFunctionNameCH = EntryFunctionNameCH.c_str();
	}
	if (!EntryFunctionNameAH.empty()) {
		HitGroupGroupDesc.hitgroup.moduleAH = Module;
		HitGroupGroupDesc.hitgroup.entryFunctionNameAH = EntryFunctionNameAH.c_str();
	}
	if (!EntryFunctionNameIS.empty()) {
		HitGroupGroupDesc.hitgroup.moduleIS = Module;
		HitGroupGroupDesc.hitgroup.entryFunctionNameIS = EntryFunctionNameIS.c_str();
	}
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		Context,
		&HitGroupGroupDesc,
		1,   // num program groups
		&GroupOpts,
		LOG, &LOG_SIZE,
		&HitGroupGroup
	));
	return HitGroupGroup;
}

OptixProgramGroup CreateMissPg(OptixDeviceContext& Context, OptixModule& Module, std::string EntryFunctionName)
{
	OptixProgramGroup MissGroup = nullptr;
	OptixProgramGroupOptions GroupOpts = {};
	OptixProgramGroupDesc MissGroupDesc = {};
	MissGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	MissGroupDesc.miss.module = Module;
	MissGroupDesc.miss.entryFunctionName = EntryFunctionName.c_str();
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		Context,
		&MissGroupDesc,
		1,   // num program groups
		&GroupOpts,
		LOG, &LOG_SIZE,
		&MissGroup
	));
	return MissGroup;
}

OptixPipeline CreatePipeline(OptixDeviceContext& Context, OptixPipelineCompileOptions PipelineCompileOpts, std::vector<OptixProgramGroup> PgUsed, uint MaxRayRecursiveDepth, uint MaxSceneTraversalDepth)
{
	return CreatePipeline(Context, PipelineCompileOpts, PgUsed.data(), PgUsed.size(), MaxRayRecursiveDepth, MaxSceneTraversalDepth);
}

OptixPipeline CreatePipeline(OptixDeviceContext& Context, OptixPipelineCompileOptions PipelineCompileOpts, OptixProgramGroup* PgUsed, uint NumOfPg, uint MaxRayRecursiveDepth, uint MaxSceneTraversalDepth)
{
	//绑定管线
	OptixPipeline Pipeline = nullptr;
	OptixPipelineLinkOptions LinkOpts = {};
	LinkOpts.maxTraceDepth = MaxRayRecursiveDepth;//先不发射光线,一层只有primary ray，改为2
	OPTIX_CHECK_LOG(optixPipelineCreate(
		Context,
		&PipelineCompileOpts,
		&LinkOpts,
		PgUsed,
		NumOfPg,
		LOG, &LOG_SIZE,
		&Pipeline
	));
	//配置栈的大小，是自动的
	OptixStackSizes StackSize = {};
	for (uint i = 0; i < NumOfPg;i++) {
		OPTIX_CHECK(optixUtilAccumulateStackSizes(PgUsed[i], &StackSize, Pipeline));
	}
	uint32_t direct_callable_stack_size_from_traversal;
	uint32_t direct_callable_stack_size_from_state;
	uint32_t continuation_stack_size;
	OPTIX_CHECK(optixUtilComputeStackSizes(&StackSize, LinkOpts.maxTraceDepth,
		0,  // maxCCDepth
		0,  // maxDCDEpth
		&direct_callable_stack_size_from_traversal,
		&direct_callable_stack_size_from_state, &continuation_stack_size));
	OPTIX_CHECK(optixPipelineSetStackSize(Pipeline, direct_callable_stack_size_from_traversal,
		direct_callable_stack_size_from_state, continuation_stack_size,
		MaxSceneTraversalDepth  // maxTraversableDepth
	));
	return Pipeline;
}


ModelData* AssembleModelData(Material mat, MyMesh& mesh, std::vector<CUdeviceptr>& GpuBuffersToRelease) {
	Material* mat_device = (Material*)UploadAnything({ &mat ,sizeof(Material) }, GpuBuffersToRelease);
	GeometryBuffer* geometry_buffer_device = CreateAndUploadGeometryBuffer(mesh, GpuBuffersToRelease);
	ModelData model_data = { geometry_buffer_device,mat_device };
	ModelData* model_data_device = (ModelData*)UploadAnything({ &model_data, sizeof(ModelData) }, GpuBuffersToRelease);
	return model_data_device;
}

void ReleaseGpuResources(std::vector<CUdeviceptr>& GpuBufferToRelease)
{
	std::sort(GpuBufferToRelease.begin(), GpuBufferToRelease.end());
	auto last = std::unique(GpuBufferToRelease.begin(), GpuBufferToRelease.end());
	GpuBufferToRelease.erase(last, GpuBufferToRelease.end());
	for (int i = 0; i < GpuBufferToRelease.size(); i++) {
		try {
			CUDA_CHECK(cudaFree((void*)GpuBufferToRelease.at(i)));
		}
		catch (std::exception& e) {
			std::cerr << "Caught exception: " << e.what() << "\t指针为:" << GpuBufferToRelease.at(i) << "\n";
		}
	}
}

CUdeviceptr UploadAnything(HostMemStream mem, std::vector<CUdeviceptr>& GpuBufferToRelease)
{
	void* Ptr;
	CUDA_CHECK(cudaMalloc(&Ptr, mem.SizeInBytes));
	CUDA_CHECK(cudaMemcpy(Ptr, mem.Ptr, mem.SizeInBytes, cudaMemcpyHostToDevice));
	GpuBufferToRelease.push_back((CUdeviceptr)Ptr);
	return (CUdeviceptr)Ptr;
}

void* UploadAnything(void* ptr, size_t size)
{
	void* Ptr;
	CUDA_CHECK(cudaMalloc(&Ptr, size));
	CUDA_CHECK(cudaMemcpy(Ptr, ptr, size, cudaMemcpyHostToDevice));
	return Ptr;
}

GeometryBuffer* CreateAndUploadGeometryBuffer(MyMesh& mesh, std::vector<CUdeviceptr>& GpuBuffersToRelease)
{
	//申请三块cpu内存
	GeometryBuffer BufferHost;
	CUDA_CHECK(cudaMalloc(BufferHost.GetNormalPPtr(), mesh.GetVerticesCount() * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(BufferHost.GetVerticesPPtr(), mesh.GetVerticesCount() * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(BufferHost.GetUvPPtr(), mesh.GetVerticesCount() * sizeof(float2)));
	//拷贝过去
	CUDA_CHECK(cudaMemcpy(BufferHost.GetNormalPtr(), mesh.GetNormalsPtr(), mesh.GetVerticesCount() * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(BufferHost.GetVerticesPtr(), mesh.GetVerticesPtr(), mesh.GetVerticesCount() * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(BufferHost.GetUvPtr(), mesh.GetUVsPtr(), mesh.GetVerticesCount() * sizeof(float2), cudaMemcpyHostToDevice));
	GeometryBuffer* BufferDevice;
	CUDA_CHECK(cudaMalloc(&BufferDevice, sizeof(GeometryBuffer)));
	CUDA_CHECK(cudaMemcpy(BufferDevice, &BufferHost, sizeof(GeometryBuffer), cudaMemcpyHostToDevice));

	GpuBuffersToRelease.push_back((CUdeviceptr)BufferHost.GetNormalPtr());
	GpuBuffersToRelease.push_back((CUdeviceptr)BufferHost.GetVerticesPtr());
	GpuBuffersToRelease.push_back((CUdeviceptr)BufferHost.GetUvPtr());
	GpuBuffersToRelease.push_back((CUdeviceptr)BufferDevice);
	return BufferDevice;
}
