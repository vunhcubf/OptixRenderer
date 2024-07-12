#include "App.h"

uint SceneManager::GetMaxRecursionDepth()
{
	return this->MaxRayRecursiveDepth;
}

void SceneManager::SetRayTracingConfig(RayTracingConfig conf)
{
	NumSbtRecords = conf.NumSbtRecords;
	pipelineCompileOptions = conf.pipelineCompileOptions;
	MaxSceneTraversalDepth = conf.MaxSceneTraversalDepth;
	MaxRayRecursiveDepth = conf.MaxRayRecursiveDepth;
}

void SceneManager::WarmUp()
{
	Context = CreateContext();
}

SceneManager::~SceneManager() {
	if (pipeLine) {
		OPTIX_CHECK(optixPipelineDestroy(pipeLine));
	}
	DestroyModules();
	DestroyShaderManager();
	if (Context)
		OPTIX_CHECK(optixDeviceContextDestroy(Context));
}

void SceneManager::DestroyShaderManager()
{
	for (auto& item : shaderManager) {
		if (item.second)
			OPTIX_CHECK(optixProgramGroupDestroy(item.second));
	}
	if (ShaderRG) {
		OPTIX_CHECK(optixProgramGroupDestroy(ShaderRG));
	}
	if (ShaderExcept) {
		OPTIX_CHECK(optixProgramGroupDestroy(ShaderExcept));
	}
	if (ShaderMiss) {
		OPTIX_CHECK(optixProgramGroupDestroy(ShaderMiss));
	}
}

void SceneManager::DestroyModules()
{
	for (auto& item : modules) {
		OPTIX_CHECK(optixModuleDestroy(item.second));
	}
}

void SceneManager::ImportCompiledShader(string path, string ModuleName)
{
	OptixModule module = CreateModule(path, Context, pipelineCompileOptions);
	modules.insert({ ModuleName,module });
	cout << "从路径: " << path << "  导入名为: " << ModuleName << endl;
}

void SceneManager::AddRayGenerationShader(string func_name, string module_name)
{//创建跟签名前创建异常处理
	ShaderExcept = nullptr;
	OptixProgramGroupOptions GroupOpts = {};//不使用payloadtype
	OptixProgramGroupDesc ExceptionGroupDesc = {};
	ExceptionGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
	ExceptionGroupDesc.exception.module = modules.at(module_name);
	ExceptionGroupDesc.exception.entryFunctionName = "__exception__shader";
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		Context,
		&ExceptionGroupDesc,
		1,   // num program groups
		&GroupOpts,
		LOG, &LOG_SIZE,
		&ShaderExcept
	));
	ShaderRG = CreateRayGenPg(Context, modules.at(module_name), func_name);
	if (ShaderRG) {
		cout << "导入光线发射函数 " << func_name << endl;
	}
}

void SceneManager::AddHitShader(string ShaderName, string module_name, string func_name_ch = "", string func_name_ah = "", string func_name_is = "")
{
	OptixProgramGroup shaderHit = CreateHitGroupPg(Context, modules.at(module_name), func_name_ch, func_name_ah, func_name_is);
	shaderManager.insert({ ShaderName,shaderHit });
	if (shaderHit) {
		cout << "导入命中函数:  CH:" << func_name_ch << "  AH:" << func_name_ah << "  IS:" << func_name_is << endl;
	}
}

void SceneManager::AddMissShader(string func_name, string module_name)
{
	ShaderMiss = CreateMissPg(Context, modules.at(module_name), func_name);
	if (ShaderMiss) {
		cout << "导入未命中函数 " << func_name << endl;
	}
}

OptixTraversableHandle SceneManager::GetTraversableHandle()
{
	return this->TASHandle;
}

void SceneManager::AddObjects(ObjectDesc desc, string Name)
{
	//构建gas
	MyMesh& mesh = desc.mesh;
	OptixTraversableHandle Handle;
	UniquePtrDevice GASOutputBuffer;
	OptixAccelBuildOptions AccelBuildOpts = {};
	AccelBuildOpts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;//构建选项，先设置为无，不压缩
	AccelBuildOpts.operation = OPTIX_BUILD_OPERATION_BUILD;
	//上传顶点缓冲
	CUdeviceptr vertexBuffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&vertexBuffer), mesh.GetVerticesCount() * sizeof(float3)));
	CUDA_CHECK(cudaMemcpy((void*)vertexBuffer, mesh.GetVerticesPtr(), mesh.GetVerticesCount() * sizeof(float3), cudaMemcpyHostToDevice));
	//构建设置
	const uint32_t TriangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
	OptixBuildInput TriangleInput = {};
	TriangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	TriangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	TriangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	TriangleInput.triangleArray.numVertices = mesh.GetVerticesCount();
	TriangleInput.triangleArray.vertexBuffers = &vertexBuffer;

	TriangleInput.triangleArray.flags = TriangleInputFlags;
	TriangleInput.triangleArray.numSbtRecords = 1;
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
		GASOutputBuffer.GetAddressOfPtr(),
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
		(CUdeviceptr)GASOutputBuffer.GetPtr(),
		GASBufferSize.outputSizeInBytes,
		&Handle,
		nullptr,            // emitted property list
		0                   // num emitted properties
	));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ScratchBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(vertexBuffer)));

	objects[Name].GASHandle = Handle;
	objects[Name].GASOutputBuffer = GASOutputBuffer;


	//构建model data
	UniquePtrDevice mat_device;
	mat_device = UploadAnything(&desc.mat, sizeof(Material));
	objects[Name].MaterialBuffer = mat_device;

	UniquePtrDevice NormalBuffer;
	UniquePtrDevice VertexBuffer;
	UniquePtrDevice TexcoordBuffer;
	CUDA_CHECK(cudaMalloc(NormalBuffer.GetAddressOfPtr(), mesh.GetVerticesCount() * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(VertexBuffer.GetAddressOfPtr(), mesh.GetVerticesCount() * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(TexcoordBuffer.GetAddressOfPtr(), mesh.GetVerticesCount() * sizeof(float2)));
	//拷贝过去
	CUDA_CHECK(cudaMemcpy(NormalBuffer.GetPtr(), mesh.GetNormalsPtr(), mesh.GetVerticesCount() * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(VertexBuffer.GetPtr(), mesh.GetVerticesPtr(), mesh.GetVerticesCount() * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(TexcoordBuffer.GetPtr(), mesh.GetUVsPtr(), mesh.GetVerticesCount() * sizeof(float2), cudaMemcpyHostToDevice));

	GeometryBuffer geobuffer_host;
	geobuffer_host.Normal = (CUdeviceptr)NormalBuffer.GetPtr();
	geobuffer_host.Vertices = (CUdeviceptr)VertexBuffer.GetPtr();
	geobuffer_host.uv = (CUdeviceptr)TexcoordBuffer.GetPtr();
	UniquePtrDevice geometryBuffer;
	geometryBuffer = UploadAnything(&geobuffer_host, sizeof(GeometryBuffer));

	objects[Name].NormalBuffer = NormalBuffer;
	objects[Name].VertexBuffer = VertexBuffer;
	objects[Name].TexcoordBuffer = TexcoordBuffer;
	objects[Name].GeometryBuffer = geometryBuffer;

	UniquePtrDevice modelData;
	ModelData modelData_host;
	modelData_host.GeometryData = (GeometryBuffer*)objects.at(Name).GeometryBuffer.GetPtr();
	modelData_host.MaterialData = (Material*)objects.at(Name).MaterialBuffer.GetPtr();
	modelData = UploadAnything(&modelData_host, sizeof(ModelData));
	objects[Name].ModelData = modelData;

	objects[Name].SbtRecordsData = vector<UniquePtrDevice>(NumSbtRecords);
	//设置sbts
	for (uint i = 0; i < NumSbtRecords; i++) {
		string& item = desc.shaders[std::min(i, NumSbtRecords)];
		objects[Name].SbtRecordsData.at(i) = (void*)CreateSbtRecord<SbtDataStruct>(shaderManager.at(item), { (CUdeviceptr)objects[Name].ModelData.GetPtr() });
	}

	cout << "\n导入物体:" << Name << endl;
	cout << "法线缓冲地址: " << objects.at(Name).NormalBuffer.GetPtr() << endl;
	cout << "顶点缓冲地址: " << objects.at(Name).VertexBuffer.GetPtr() << endl;
	cout << "纹理坐标缓冲地址: " << objects.at(Name).TexcoordBuffer.GetPtr() << endl;
	cout << "几何缓冲地址: " << objects.at(Name).GeometryBuffer.GetPtr() << endl;
	cout << "材质缓冲地址: " << objects.at(Name).MaterialBuffer.GetPtr() << endl;
	cout << "模型缓冲地址: " << objects.at(Name).ModelData.GetPtr() << endl;
	cout << "GAS句柄: " << objects.at(Name).GASHandle << endl;
	for (uint i = 0; i < NumSbtRecords; i++) {
		cout << "SBT记录地址" << i << ": " << objects.at(Name).SbtRecordsData.at(i).GetPtr() << endl;
	}

}

void SceneManager::ConfigureMissSbt(MissData Data)
{
	SbtRecordMiss = (void*)CreateSbtRecord<MissData>(ShaderMiss, Data);
}

void SceneManager::ConfigureRGSbt(RayGenData Data)
{
	SbtRecordRG = (void*)CreateSbtRecord<RayGenData>(ShaderRG, Data);
}

void SceneManager::BuildScene()
{
	cout << "\nCall Build Scene" << endl;
	uint NumObjs = objects.size();
	vector<OptixInstance> instances(NumObjs);
	float TransformIdentity[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
	ZeroMemory(instances.data(), sizeof(OptixInstance) * NumObjs);

	//任务是创建TAS和设置SBT
	CUDA_CHECK(cudaMalloc(SbtRecordHit.GetAddressOfPtr(), NumObjs * NumSbtRecords * sizeof(SbtRecord<SbtDataStruct>)));
	uint Counter = 0;
	cout << "SBT记录大小: " << sizeof(SbtRecord<SbtDataStruct>) << endl;
	for (auto& item : objects) {
		Object& obj = item.second;
		const string& obj_name = item.first;

		instances[Counter].instanceId = Counter;
		instances[Counter].sbtOffset = Counter * NumSbtRecords;
		instances[Counter].visibilityMask = 255;
		instances[Counter].traversableHandle = obj.GASHandle;
		for (int i = 0; i < 12; i++) {
			instances[Counter].transform[i] = TransformIdentity[i];
		}
		for (uint i = 0; i < NumSbtRecords; i++) {
			SbtRecord<SbtDataStruct>* OffsetAddress = (SbtRecord<SbtDataStruct>*)SbtRecordHit.GetPtr();
			OffsetAddress += Counter * NumSbtRecords + i;
			cout << "偏移量: " << OffsetAddress << endl;
			CUDA_CHECK(cudaMemcpy(OffsetAddress, obj.SbtRecordsData[i].GetPtr(), sizeof(SbtRecord<SbtDataStruct>), cudaMemcpyDeviceToDevice));
		}
		Counter++;
	}
	//装配hitgroup sbtrecords
	OptixInstance* InstancesDevice;
	CUDA_CHECK(cudaMalloc(&InstancesDevice, sizeof(OptixInstance) * NumObjs));
	CUDA_CHECK(cudaMemcpy(InstancesDevice, instances.data(), sizeof(OptixInstance) * NumObjs, cudaMemcpyHostToDevice));
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixBuildInput InstanceInput = {};
	InstanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	InstanceInput.instanceArray.instances = (CUdeviceptr)InstancesDevice;
	InstanceInput.instanceArray.numInstances = NumObjs;

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
		this->IASOutputBuffer.GetAddressOfPtr(),
		IASBufferSize.outputSizeInBytes
	));
	OPTIX_CHECK(optixAccelBuild(
		Context,
		0,                  // CUDA stream
		&accelOptions,
		&InstanceInput,
		1,                  // num build inputs
		ScratchBuffer,
		IASBufferSize.tempSizeInBytes,
		(CUdeviceptr)this->IASOutputBuffer.GetPtr(),
		IASBufferSize.outputSizeInBytes,
		&this->TASHandle,
		nullptr,            // emitted property list
		0                   // num emitted properties
	));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ScratchBuffer)));
	cout << "TAS句柄: " << TASHandle << endl;
	SbtRecordException = (void*)CreateSbtRecord<CUdeviceptr>(ShaderRG, 0);
	OptixShaderBindingTable& ShaderBindingTable = Sbt;
	ShaderBindingTable.raygenRecord = (CUdeviceptr)SbtRecordRG.GetPtr();;
	ShaderBindingTable.missRecordBase = (CUdeviceptr)SbtRecordMiss.GetPtr();
	ShaderBindingTable.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
	ShaderBindingTable.missRecordCount = 1;
	ShaderBindingTable.hitgroupRecordBase = (CUdeviceptr)SbtRecordHit.GetPtr();
	ShaderBindingTable.hitgroupRecordStrideInBytes = sizeof(SbtRecord<SbtDataStruct>);
	ShaderBindingTable.hitgroupRecordCount = NumObjs * NumSbtRecords;
	ShaderBindingTable.exceptionRecord = (CUdeviceptr)SbtRecordException.GetPtr();

	vector<OptixProgramGroup> shaders_array;
	shaders_array.reserve(shaderManager.size() + 2);
	for (auto& item : shaderManager) {
		shaders_array.push_back(item.second);
	}
	shaders_array.push_back(this->ShaderMiss);
	shaders_array.push_back(this->ShaderRG);
	pipeLine = CreatePipeline(Context, pipelineCompileOptions, shaders_array,MaxRayRecursiveDepth,MaxSceneTraversalDepth);
}

void SceneManager::DispatchRays(uchar4* FrameBuffer, CUstream& Stream, LaunchParametersDesc LParamsDesc, uint Width, uint Height, uint Spp)
{
	LaunchParameters params;
	params.ImagePtr = FrameBuffer;
	params.Width = Width;
	params.Height = Height;
	params.Handle = this->TASHandle;
	params.Seed = static_cast<uint>(rand());
	params.cameraData = LParamsDesc.cameraData;
	params.areaLight = LParamsDesc.areaLight;
	params.Spp = Spp;
	params.MaxRecursionDepth = MaxRayRecursiveDepth;
	LaunchParameter = UploadAnything(&params, sizeof(LaunchParameters));
	OPTIX_CHECK(optixLaunch(this->pipeLine, Stream, (CUdeviceptr)LaunchParameter.GetPtr(), sizeof(LaunchParameters), &Sbt, Width, Height, 1));
	CUDA_CHECK(cudaStreamSynchronize(Stream));
}
void SceneManager::DispatchRays(uchar4* FrameBuffer, CUstream& Stream, LaunchParameters* LParams, uint Width, uint Height, uint Spp)
{
	OPTIX_CHECK(optixLaunch(this->pipeLine, Stream, (CUdeviceptr)LParams, sizeof(LaunchParameters), &Sbt, Width, Height, 1));
	CUDA_CHECK(cudaStreamSynchronize(Stream));
}
//WASDEQ--
void MyCamera::Update(int64 DeltaTIme,
	uint KeyBoardActionBitMask,
	uint MouseActionBitMask,
	double2 MousePos,
	double2 MousePosPrev)
{
	if (!GetState(MouseActionBitMask, INPUT_TYPE::MOUSE_RIGHT)) {
		return;
	}
	float Speed = 4e-3f;
	int DirectionWS = GetState(KeyBoardActionBitMask, INPUT_TYPE::W) - GetState(KeyBoardActionBitMask, INPUT_TYPE::S);
	int DirectionAD = GetState(KeyBoardActionBitMask, INPUT_TYPE::D) - GetState(KeyBoardActionBitMask, INPUT_TYPE::A);
	int DirectionQE = GetState(KeyBoardActionBitMask, INPUT_TYPE::E) - GetState(KeyBoardActionBitMask, INPUT_TYPE::Q);

	DirectionAD = -DirectionAD;
	DirectionQE = -DirectionQE;
	//根据现有的坐标更新位置
	float3& Pos = this->WorldPos;
	float3 Movement = make_float3(
		ForwardDirection.x * Speed * DirectionWS * DeltaTIme + RightDirection.x * Speed * DirectionAD * DeltaTIme + UpDirection.x * Speed * DirectionQE * DeltaTIme,
		ForwardDirection.y * Speed * DirectionWS * DeltaTIme + RightDirection.y * Speed * DirectionAD * DeltaTIme + UpDirection.y * Speed * DirectionQE * DeltaTIme,
		ForwardDirection.z * Speed * DirectionWS * DeltaTIme + RightDirection.z * Speed * DirectionAD * DeltaTIme + UpDirection.z * Speed * DirectionQE * DeltaTIme
	);
	Pos = float3_add(Pos, Movement);
	//更新完位置再更新旋转
	float Speed_rotate = 4e-4f;
	double2 MouseDelta = make_double2(MousePos.x - MousePosPrev.x, MousePos.y - MousePosPrev.y);
	mPhi += MouseDelta.y * Speed_rotate * DeltaTIme;
	mTheta -= MouseDelta.x * Speed_rotate * DeltaTIme;
	mPhi = std::clamp(mPhi, M_PI * 0.05f, M_PI * 0.95f);
	while (mTheta > 2 * M_PI) {
		mTheta -= 2 * M_PI;
	}
	//更新旋转
	ForwardDirection = make_float3(sinf(mPhi) * cosf(mTheta), sinf(mPhi) * sinf(mTheta), cosf(mPhi));
	//由于不可能歪头
	RightDirection = make_float3(cosf(mTheta + M_PI / 2), sinf(mTheta + M_PI / 2), 0);
	UpDirection = CrossProduct(RightDirection, ForwardDirection);
}

CameraData MyCamera::ExportCameraData(int width, int height)
{
	CameraData Data;
	Data.cam_eye = this->WorldPos;
	float aspect = (float)width / (float)height;
	Data.cam_v = float3_scale(UpDirection, std::tan(Fov / 2));
	Data.cam_u = float3_scale(RightDirection, std::tan(Fov / 2) * aspect);
	Data.cam_w = ForwardDirection;
	return Data;
}
MyCamera::MyCamera(float3 Pos, float2 Rotate) {
	this->WorldPos = Pos;
	this->mPhi = Rotate.y;
	this->mTheta = Rotate.x;
	ForwardDirection = make_float3(sinf(mPhi) * cosf(mTheta), sinf(mPhi) * sinf(mTheta), cosf(mPhi));
	//由于不可能歪头
	RightDirection = make_float3(cosf(mTheta + M_PI / 2), sinf(mTheta + M_PI / 2), 0);
	UpDirection = CrossProduct(RightDirection, ForwardDirection);
}


