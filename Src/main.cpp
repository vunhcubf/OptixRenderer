#include <sutil.h>
#include <ctime>
#include "App.h"
#include "Texture.h"
#include "objloader.h"
#include <bitset>
#include <GLDisplay.h>
#include <atomic>
#include "runtime_shader_compile.h"
#include "random_host.h"
#include <windows.h>
#include "ComputeShader.h"
inline std::string getExecutablePath() {
    char path[MAX_PATH];
    if (GetModuleFileNameA(NULL, path, MAX_PATH) > 0) {
        return std::string(path);
    }
    return "";
}
inline std::string getParentDir(std::string path) {
    std::filesystem::path filepath=path;
    return filepath.parent_path().string();
}
const uint default_width = 1024;
const uint default_height = 1024;
using namespace sutil;
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec2 aPos;\n"
"layout (location = 1) in vec2 aTexCoords;\n"

"out vec2 TexCoords;\n"

"void main() {\n"
"gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
"TexCoords = aTexCoords;\n"
"}\n\0";
const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"

"in vec2 TexCoords;\n"

"uniform sampler2D textureSampler;\n"

"void main() {\n"
"FragColor = texture(textureSampler, TexCoords);\n"
"}\n\0";
TextureManager* TextureManager::instance = nullptr;
void main() {
	try {
		string CUDAIncludePath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include";
		string OptixIncludePath = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include";
		string ProjectPath = getParentDir(getParentDir(getParentDir(getExecutablePath())));// 向前滚动两级别
		string CompiledShaderPath = ProjectPath + "/CompiledShaders";
		string ShaderPath = ProjectPath + "/Shaders";
		string BlueNoiseMapPath=ProjectPath + "/Assets/Textures/stbn_vec3_2Dx1D_128x128x64_49.png";
		// 加载蓝噪声图
		BlueNoiseMapBufferManager BlueNoise(BlueNoiseMapPath.c_str());
		// 所有文件都编译
		ShaderCollection shader_sources = ReadShaderSources(ProjectPath);
		// 先检测有哪些shader，对于存在的shader比较hash值，对于新的shader直接加入编译
		// 开始编译
		{
			for (auto& shader_to_compile_inst : shader_sources) {
				std::cout << "正在编译着色器：" << shader_to_compile_inst.first << endl;
				nvrtcProgram prog;
				nvrtcCreateProgram(&prog,             // 程序
					shader_to_compile_inst.second.c_str(),    // 源码字符串
					shader_to_compile_inst.first.c_str(),    // 内核名称
					0,
					nullptr,
					nullptr);    // 头文件和包括
		
				// 编译选项，指定 GPU 架构
				const char* opts[] = { 
					"--gpu-architecture=compute_89",
					"-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include",
					"-IC:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include",
					"-IC:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/SDK",
					"-ID:/OptixRenderer/ShaderLibrary",
				};
		
				nvrtcResult compileResult = nvrtcCompileProgram(prog, 5, opts);
		
				// 检查编译错误
				if (compileResult != NVRTC_SUCCESS) {
					size_t logSize;
					nvrtcGetProgramLogSize(prog, &logSize);
					std::vector<char> log(logSize);
					nvrtcGetProgramLog(prog, log.data());
					std::cerr << "Compilation error:"<<compileResult<<"\n" << log.data() << std::endl;
				}
		
				// 获取 PTX 代码
				size_t ptxSize;
				nvrtcGetPTXSize(prog, &ptxSize);
				std::vector<char> ptx(ptxSize);
				nvrtcGetPTX(prog, ptx.data());
				// 写入PTX
				std::ofstream file(CompiledShaderPath+"/"+ shader_to_compile_inst.first+".ptx");
		
				if (!file) {
					std::cerr << "Error opening file for writing: " << shader_to_compile_inst.first<<".ptx" << std::endl;
					return;
				}
		
				file << ptx.data();
		
				file.close();
			}
		}
		Texture2D skybox = Texture2D::LoadImageFromFile(ProjectPath + "/Assets/Textures/kloofendal_48d_partly_cloudy_puresky_2k.exr");
		//uint64 SkyBoxTex=TextureManager::Add(Texture2D::LoadImageFromFile(ProjectPath + "/Assets/Textures/kloofendal_48d_partly_cloudy_puresky_2k.exr"));
		//uint64 SkyBoxTex2=TextureManager::Add(Texture2D::LoadImageFromFile(ProjectPath + "/Assets/Textures/citrus_orchard_road_2k.exr"));
		//uint64 SkyBoxTex3=TextureManager::Add(Texture2D::LoadImageFromFile(ProjectPath + "/Assets/Textures/autumn_field_puresky_2k.exr"));
		//uint64 SkyBoxStudio=TextureManager::Add(Texture2D::LoadImageFromFile(ProjectPath + "/Assets/Textures/studio_small_05_2k.exr"));
		
		uint KeyBoardActionBitMask = 0U;
		uint MouseActionBitMask = 0U;
		double2 MousePos;
		double2 MousePosPrev;
		//配置场景
		SceneManager scene;
		scene.WarmUp();
		RayTracingConfig conf;
		conf.NumSbtRecords = 2;
		conf.MaxRayRecursiveDepth = 10;
		conf.MaxSceneTraversalDepth = 2;
		conf.pipelineCompileOptions = CreatePipelineCompileOptions(OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY, 16, 2);
		scene.SetRayTracingConfig(conf);

		scene.ImportCompiledShader(ProjectPath + "/CompiledShaders/smallpt_styled.cu.ptx", "module_smallpt_styled");
		scene.ImportCompiledShader(ProjectPath + "/CompiledShaders/basic.cu.ptx", "module_basic");
		scene.ImportCompiledShader(ProjectPath + "/CompiledShaders/disney_principled_pt.cu.ptx", "module_disney_principled");
		// 加载一般的核函数
		string AccumulateCsPath = ProjectPath + "/CompiledShaders/kernels.cu.ptx";

		CUmodule AccumulateCs = LoadModule(AccumulateCsPath);
		CUfunction AccumulateCs_Fn = LoadFunction(AccumulateCs, "AccumulateFrame");

		scene.AddMissShader("__miss__fetchMissInfo", "module_disney_principled");
		scene.AddRayGenerationShader("__raygen__principled_bsdf", "module_disney_principled");
		scene.AddHitShader("CH_diffuse", "module_smallpt_styled", "__closesthit__diffuse", "", "");
		scene.AddHitShader("CH_glossy", "module_smallpt_styled", "__closesthit__glossy", "", "");
		scene.AddHitShader("CH_glass", "module_smallpt_styled", "__closesthit__glass", "", "");
		scene.AddHitShader("CH_occluded", "module_smallpt_styled", "__closesthit__occluded", "", "");

		scene.AddHitShader("CH_principled_bsdf", "module_disney_principled", "__closesthit__principled_bsdf", "", "");

		scene.AddHitShader("CH_fetchHitInfo", "module_disney_principled", "__closesthit__fetch_hitinfo", "", "");

		{
			ObjLoadResult cornel = LoadObj(ProjectPath + "/Assets/Models/cornel.obj");
			for (const auto& one : cornel) {
				const string& name = one.first;
				const Mesh& mesh = one.second.first;
				Material mat = one.second.second;
				ObjectDesc desc;
				desc.mesh = mesh;
				desc.mat = mat;
				desc.shaders = { "CH_fetchHitInfo","CH_fetchHitInfo" };
				scene.AddObjects(desc, name);
			}
		}
		{
			ObjLoadResult sponza = LoadObj(ProjectPath + "/Assets/Models/Sponza/sponza.obj");
			for (const auto& one : sponza) {
				const string& name = one.first;
				const Mesh& mesh = one.second.first;
				const Material& mat = one.second.second;
				ObjectDesc desc;
				desc.mesh = mesh;
				desc.mat = mat;
				desc.shaders = { "CH_fetchHitInfo","CH_fetchHitInfo" };
				scene.AddObjects(desc, name);
			}
		}
		{string name = "area_light";
		ObjectDesc desc;
		desc.mesh = Mesh::LoadMeshFromFile(ProjectPath + "/Assets/Models/" + name + ".obj");
		desc.mat.MaterialType = MATERIAL_AREALIGHT;
		desc.shaders = { "CH_fetchHitInfo","CH_fetchHitInfo" };
		scene.AddObjects(desc, name); }

		//scene.ConfigureMissSbt({ make_float3(0,0,0),1.0f,TextureManager::QueryTex2DWithIndex(SkyBoxTex2)->GetTextureView().textureIdentifier});
		scene.ConfigureMissSbt({ make_float3(0,0,0),1.0f,skybox.GetTextureView().textureIdentifier });
		scene.ConfigureRGSbt({ 1.0f,0.0f,1.0f });
		scene.BuildScene();

		MyCamera camera(make_float3(0, -5.0f, 0), make_float2(M_PI * 0.5, M_PI / 2));

		AreaLight Light;
		Light.P1 = make_float3(-0.384588, -0.384588, 1.324847);
		Light.P2 = make_float3(0.384588, -0.384588, 1.324847);
		Light.P4 = make_float3(-0.384588, 0.384588, 1.324847);
		Light.P3 = make_float3(0.384588, 0.384588, 1.324847);
		Light.Color = make_float3(20,20,20);
		Light.Area = 0.652511;
		LaunchParametersDesc desc;
		desc.areaLight = Light;
		desc.cameraData = camera.ExportCameraData(default_width, default_height);

		//UniquePtrDevice textureViewsDevice;
		//CUDA_CHECK(cudaMalloc(textureViewsDevice.GetAddressOfPtr(),sizeof(TextureManager)*TextureManager::GetSize()));
		//CUDA_CHECK(cudaMemcpy(textureViewsDevice.GetPtr(),));

		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		// glfw window creation
		// --------------------
		GLFWwindow* window = glfwCreateWindow(default_width, default_height, "OptixRenderer", NULL, NULL);
		glfwMakeContextCurrent(window);
		glfwGetCursorPos(window, &MousePos.x, &MousePos.y);
		glfwGetCursorPos(window, &MousePosPrev.x, &MousePosPrev.y);
		// glad: load all OpenGL function pointers
		// ---------------------------------------
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
			return;
		}


		// build and compile our shader program
		// ------------------------------------
		// vertex shader
		unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
		glCompileShader(vertexShader);
		// check for shader compile errors
		int success;
		char infoLog[512];
		glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		// fragment shader
		unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
		glCompileShader(fragmentShader);
		// check for shader compile errors
		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		// link shaders
		unsigned int shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);
		// check for linking errors
		glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		}
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		// set up vertex data (and buffer(s)) and configure vertex attributes
		// ------------------------------------------------------------------
		float vertices[] = {
		-1.0f, -1.0f,  1.0f, 1.0f,
		 1.0f, -1.0f,  0.0f, 1.0f,
		 1.0f,  1.0f,  0.0f, 0.0f,

		 -1.0f, -1.0f,  1.0f, 1.0f,
		 -1.0f,  1.0f,  1.0f, 0.0f,
		 1.0f,  1.0f,  0.0f, 0.0f,
		};

		GLuint VAO, VBO;
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);



		// uncomment this call to draw in wireframe polygons.
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		GLuint pbo;
		glGenBuffers(1, &pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, default_width * default_height * 4, NULL, GL_STREAM_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, default_width, default_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		cudaGraphicsResource* cuda_pbo_resource;
		cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
		CUstream Stream;
		CUDA_CHECK(cudaStreamCreate(&Stream));
		// render loop
		// -----------
		int prev_width = default_width;
		int prev_height = default_height;
		//游戏计时器
		using namespace std::chrono;
		auto start = high_resolution_clock::now();
		int64 delta_time = 0;
		//上一帧的鼠标位置
		//相机和灯光选项
		UniquePtrDevice LParams;
		CUDA_CHECK(cudaMalloc(LParams.GetAddressOfPtr(), sizeof(LaunchParameters)));
		bool FirstTime = true;
		//帧计数器
		uint64 FrameCounter = 0;
		//相机数据在lparams中的地址偏移
		LaunchParameters* p = nullptr;
		uint64 AddressBiasOfFrameNumberInLaunchParams = (uint64)(&p->FrameNumber) - (uint64)(p);
		uint64 AddressBiasOfCameraDataInLaunchParams = (uint64)(&p->cameraData) - (uint64)(p);
		uint64 AddressBiasOfSeedInLaunchParams = (uint64)(&p->Seed) - (uint64)(p);
		uint64 AddressBiasOfHeightInLaunchParams = (uint64)(&p->Height) - (uint64)(p);
		uint64 AddressBiasOfWidthInLaunchParams = (uint64)(&p->Width) - (uint64)(p);
		uint64 AddressBiasOfPixelOffsetInLaunchParams = (uint64)(&p->PixelOffset) - (uint64)(p);
		uint64 AddressBiasOfIndirectOutputBufferInLaunchParams = (uint64)(&p->IndirectOutputBuffer) - (uint64)(p);
		//创建渲染纹理用于帧结果累计
		//希望渲染纹理不要频繁申请释放
		const uint expected_max_width = (uint)GetSystemMetrics(SM_CXSCREEN);
		const uint expected_max_height = (uint)GetSystemMetrics(SM_CYSCREEN);
		UniquePtrDevice FrameBuffer2, FrameBuffer3;
		CUDA_CHECK(cudaMalloc(FrameBuffer2.GetAddressOfPtr(), sizeof(float3) * expected_max_width * expected_max_height));
		CUDA_CHECK(cudaMalloc(FrameBuffer3.GetAddressOfPtr(), sizeof(float3) * expected_max_width * expected_max_height));
		uint64 FrameNumber = 0;

		// 初始化随机数器
		uint64* RandomGeneratorPixelOffset;
		{
			uint PixelCount=prev_height*prev_width;
			CUDA_CHECK(cudaMalloc(&RandomGeneratorPixelOffset,sizeof(uint64)*PixelCount));
			CUDA_CHECK(cudaMemset(RandomGeneratorPixelOffset,0,sizeof(uint64)*PixelCount));
		}

		while (!glfwWindowShouldClose(window))
		{
			int width, height;
			glfwGetWindowSize(window, &width, &height);
			glViewport(0, 0, width, height);
			if (prev_width != width || prev_height != height) {
				prev_width = width;
				prev_height = height;
				//先删除资源
				cudaGraphicsUnregisterResource(cuda_pbo_resource);
				glDeleteBuffers(1, &pbo);
				glDeleteTextures(1, &texture);
				//重新创建资源
				glGenBuffers(1, &pbo);
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
				glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_STREAM_DRAW);
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

				glGenTextures(1, &texture);
				glBindTexture(GL_TEXTURE_2D, texture);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

				cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
				//重新设置launchparams
				uint64 BiasedAddress = (uint64)LParams.GetPtr() + AddressBiasOfWidthInLaunchParams;
				uint temp;
				temp = width;
				CUDA_CHECK(cudaMemcpyAsync((void*)BiasedAddress, &temp, sizeof(uint), cudaMemcpyHostToDevice, Stream));
				temp = height;
				BiasedAddress = (uint64)LParams.GetPtr() + AddressBiasOfHeightInLaunchParams;
				CUDA_CHECK(cudaMemcpyAsync((void*)BiasedAddress, &temp, sizeof(uint), cudaMemcpyHostToDevice, Stream));
				//重新设置相机数据
				BiasedAddress = (uint64)LParams.GetPtr() + AddressBiasOfCameraDataInLaunchParams;
				CameraData data = camera.ExportCameraData(width, height);
				CUDA_CHECK(cudaMemcpyAsync((void*)BiasedAddress, &data, sizeof(CameraData), cudaMemcpyHostToDevice, Stream));
				// 重新设置像素累计数据
				{		
					uint64 BiasedAddress = (uint64)LParams.GetPtr() + AddressBiasOfPixelOffsetInLaunchParams;		
					CUDA_CHECK(cudaFree(RandomGeneratorPixelOffset));
					uint PixelCount=prev_height*prev_width;
					CUDA_CHECK(cudaMalloc(&RandomGeneratorPixelOffset,sizeof(uint64)*PixelCount));
					CUDA_CHECK(cudaMemset(RandomGeneratorPixelOffset,0,sizeof(uint64)*PixelCount));
					CUDA_CHECK(cudaMemcpy((void*)BiasedAddress,&RandomGeneratorPixelOffset,sizeof(uint64*),cudaMemcpyHostToDevice));
				}
				//重新创建FrameBuffer2
				//重新累计
				FrameCounter = 0;
				if (width > expected_max_width || height > expected_max_height) {
					CUDA_CHECK(cudaFree(FrameBuffer2.GetPtr()));
					CUDA_CHECK(cudaMalloc(FrameBuffer2.GetAddressOfPtr(), sizeof(float3) * width * height));
					CUDA_CHECK(cudaFree(FrameBuffer3.GetPtr()));
					CUDA_CHECK(cudaMalloc(FrameBuffer3.GetAddressOfPtr(), sizeof(float3) * width * height));
					//更新lparams中的参数
					CUdeviceptr temp = (CUdeviceptr)FrameBuffer2.GetPtr();
					CUDA_CHECK(cudaMemcpyAsync((void*)((uint64)LParams.GetPtr() + AddressBiasOfIndirectOutputBufferInLaunchParams), &temp, sizeof(CUdeviceptr), cudaMemcpyHostToDevice, Stream));
				}
			}
			//处理键鼠输入
			SetState(KeyBoardActionBitMask, INPUT_TYPE::W, glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS);
			SetState(KeyBoardActionBitMask, INPUT_TYPE::A, glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS);
			SetState(KeyBoardActionBitMask, INPUT_TYPE::S, glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS);
			SetState(KeyBoardActionBitMask, INPUT_TYPE::D, glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS);
			SetState(KeyBoardActionBitMask, INPUT_TYPE::Q, glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS);
			SetState(KeyBoardActionBitMask, INPUT_TYPE::E, glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS);

			MousePosPrev = MousePos;
			glfwGetCursorPos(window, &MousePos.x, &MousePos.y);

			SetState(MouseActionBitMask, INPUT_TYPE::MOUSE_LEFT, glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
			SetState(MouseActionBitMask, INPUT_TYPE::MOUSE_RIGHT, glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

			cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
			uchar4* devPtr;
			size_t size;
			cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cuda_pbo_resource);
			if (FirstTime) {
				FirstTime = false;
				LaunchParameters params;
				params.ImagePtr = devPtr;
				params.Width = width;
				params.Height = height;
				params.Handle = scene.GetTraversableHandle();
				params.Seed = static_cast<uint>(rand());
				params.cameraData = camera.ExportCameraData(width, height);
				params.areaLight = Light;
				params.Spp = 1;
				params.MaxRecursionDepth = scene.GetMaxRecursionDepth();
				params.IndirectOutputBuffer = (float3*)FrameBuffer2.GetPtr();
				params.PixelOffset=RandomGeneratorPixelOffset;
				params.BlueNoiseBuffer=BlueNoise.GetBuffer();
				CUDA_CHECK(cudaMemcpyAsync(LParams.GetPtr(), &params, sizeof(LaunchParameters), cudaMemcpyHostToDevice, Stream));
			}

			//开始渲染逻辑
			//更新lparams
			if (KeyBoardActionBitMask || MouseActionBitMask) {
				FrameCounter = 0;
				camera.Update(delta_time,
					KeyBoardActionBitMask,
					MouseActionBitMask,
					MousePos,
					MousePosPrev);
				//只拷贝相机数据
				CameraData data = camera.ExportCameraData(width, height);
				uint64 BiasedAddressOfParams = (uint64)LParams.GetPtr() + AddressBiasOfCameraDataInLaunchParams;
				CUDA_CHECK(cudaMemcpyAsync((void*)BiasedAddressOfParams, &data, sizeof(CameraData), cudaMemcpyHostToDevice, Stream));
			}
			//拷贝随机种
			uint64 BiasedAddressOfParams = (uint64)LParams.GetPtr() + AddressBiasOfSeedInLaunchParams;
			uint seed = static_cast<uint>(rand());
			CUDA_CHECK(cudaMemcpyAsync((void*)BiasedAddressOfParams, &seed, sizeof(uint), cudaMemcpyHostToDevice, Stream));
			CUDA_CHECK(cudaMemcpyAsync((void*)((uint64)LParams.GetPtr() + AddressBiasOfFrameNumberInLaunchParams), &FrameNumber, sizeof(uint64), cudaMemcpyHostToDevice, Stream));
			CUDA_CHECK(cudaGetLastError());
			scene.DispatchRays(devPtr, Stream, (LaunchParameters*)LParams.GetPtr(), width, height);

			//希望当相机不动时累计结果
			{
				uint pixel_count = width * height;
				void* args[] = { &pixel_count,&FrameCounter,&devPtr,FrameBuffer2.GetAddressOfPtr(),FrameBuffer3.GetAddressOfPtr()};

				uint threads_per_block = (uint)min(512, width * height);
				uint num_blocks = (uint)ceil(width * height / (float)threads_per_block);

				cuLaunchKernel(AccumulateCs_Fn,
					num_blocks, 1, 1,              // Grid dimension
					threads_per_block, 1, 1,            // Block dimension
					0,                                // Shared memory size
					Stream,                         // Stream
					args,                             // Kernel arguments
					nullptr);
				cuStreamSynchronize(Stream);
				FrameCounter++;
			}

			CUDA_CHECK(cudaStreamSynchronize(Stream));
			CUDA_CHECK(cudaGetLastError());
			cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

			// 使用该纹理渲染场景（例如，绘制一个覆盖整个视口的四边形）

			// render
			// ------
			glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);

			// draw our first triangle
			glUseProgram(shaderProgram);
			glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
			glBindTexture(GL_TEXTURE_2D, texture);
			glDrawArrays(GL_TRIANGLES, 0, 6);
			// glBindVertexArray(0); // no need to unbind it every time 

			// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
			// -------------------------------------------------------------------------------
			glfwSwapBuffers(window);
			glfwPollEvents();
			//更新游戏计数器
			auto end = high_resolution_clock::now();
			delta_time = duration_cast<milliseconds>(end - start).count();
			start = end;
			stringstream ss;
			ss << "Optix Renderer     FPS:" << 1000.0f / delta_time << endl;
			glfwSetWindowTitle(window, ss.str().c_str());
			FrameNumber++;
		}
		CUDA_CHECK(cudaFree(RandomGeneratorPixelOffset));
		// optional: de-allocate all resources once they've outlived their purpose:
		// ------------------------------------------------------------------------
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteProgram(shaderProgram);

		// glfw: terminate, clearing all previously allocated GLFW resources.
		// ------------------------------------------------------------------
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &texture);
		glfwDestroyWindow(window);
		glfwTerminate();

		TextureManager::DeleteInstance();
		return;
	}
	catch (std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << "\n";
		system("pause");
	}
}