#include <sutil.h>
#include <ctime>
#include "App.h"
#include <bitset>
#include <GLDisplay.h>
#include <atomic>
#include "runtime_shader_compile.h"
#include "random_host.h"
#include <windows.h>
#include "Light.h"
#include "EMSample.h"
#include "EMSample.cuh"
#include <glad/glad.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#include < GLFW/glfw3native.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define DEBUG_BUFF_SIZE sizeof(float3) * 128
const char* debugModeItems[] = {
	"NoDebug", 
	"PrimaryRayHitObject",
	"FirstBsdfRayHitObject",
	"SecondBsdfRayHitObject",
	"ThirdBsdfRayHitObject",

	"FirstNEERayHitObject",
	"SecondNEERayHitObject",
	"ThirdNEERayHitObject",

	"FinalBsdfRayHitObject",
	"FinalRadianceIndirect",
	"FinalRadianceDirect",
	"FinalWeight",
	"FinalWeightClip",

	"DebugLightPath",
	"DebugGlobalLightPath"
};
const char* frameAccumulationItems[] = {
	"ForceOn", "ForceOff", "Auto"
};
bool DepthTestWhenDebugLightPath = false;
extern "C" 	void LaunchDrawWorldLineParallel(
	uchar4 * framebuffer,
	const float* depthBuffer,
	int width,
	int height,

	float3 cam_eye,
	float3 cam_u,
	float3 cam_v,
	float3 cam_w,

	float3 p0World,
	float3 p1World,

	uchar4 color,
	float lineWidthPixels,
	bool DepthTestWhenDebugLightPath
);
using namespace sutil;
static const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec2 aPos;\n"
"layout (location = 1) in vec2 aTexCoords;\n"

"out vec2 TexCoords;\n"

"void main() {\n"
"gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
"TexCoords = aTexCoords;\n"
"}\n\0";
static const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"

"in vec2 TexCoords;\n"

"uniform sampler2D textureSampler;\n"

"void main() {\n"
"FragColor = texture(textureSampler, TexCoords);\n"
"}\n\0";
void disableMinimizeButton(GLFWwindow* window) {
	HWND hwnd = glfwGetWin32Window(window); // 获取 Win32 窗口句柄
	LONG style = GetWindowLong(hwnd, GWL_STYLE); // 获取当前窗口样式
	style &= ~WS_MINIMIZEBOX; // 移除最小化按钮样式
	SetWindowLong(hwnd, GWL_STYLE, style); // 应用新的样式
	SetWindowPos(hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED); // 刷新窗口
}

void main() {
	try {
		string CUDAIncludePath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include";
		string OptixIncludePath = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include";
		string ProjectPath = getParentDir(getParentDir(getParentDir(getExecutablePath())));// 向前滚动两级别
		string CompiledShaderPath = ProjectPath + "/CompiledShaders";
		string ShaderPath = ProjectPath + "/Shaders";
		// 所有文件都编译
		ShaderCollection shader_sources = ReadShaderSources(ProjectPath);
		// 先检测有哪些shader，对于存在的shader比较hash值，对于新的shader直接加入编译
		// 开始编译
		stringstream compilationOutput;
		{

			for (auto& shader_to_compile_inst : shader_sources) {
				compilationOutput << "正在编译着色器：" << shader_to_compile_inst.first << "\n";
				nvrtcProgram prog;
				nvrtcCreateProgram(&prog,             // 程序
					shader_to_compile_inst.second.c_str(),    // 源码字符串
					shader_to_compile_inst.first.c_str(),    // 内核名称
					0,
					nullptr,
					nullptr);    // 头文件和包括

				// 编译选项，指定 GPU 架构
				const char* opts[] = {
					"--prec-sqrt=true",
					"--prec-div=true",
					"-ftz=false",
					"--gpu-architecture=compute_89",
					"-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include",
					"-IC:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include",
					"-IC:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/SDK",
					"-ID:/OptixRenderer/ShaderLibrary"
				};

				nvrtcResult compileResult = nvrtcCompileProgram(prog, sizeof(opts)/sizeof(opts[0]), opts);

				// 检查编译错误
				if (compileResult != NVRTC_SUCCESS) {
					size_t logSize;
					nvrtcGetProgramLogSize(prog, &logSize);
					std::vector<char> log(logSize);
					nvrtcGetProgramLog(prog, log.data());
					compilationOutput << "Compilation error:" << compileResult << "\n" << log.data() << "\n";
					//std::cerr << "Compilation error:"<<compileResult<<"\n" << log.data() << std::endl;
				}

				// 获取 PTX 代码
				size_t ptxSize;
				nvrtcGetPTXSize(prog, &ptxSize);
				std::vector<char> ptx(ptxSize);
				nvrtcGetPTX(prog, ptx.data());
				// 写入PTX
				std::ofstream file(CompiledShaderPath + "/" + shader_to_compile_inst.first + ".ptx");

				if (!file) {
					compilationOutput << "Error opening file for writing: " << shader_to_compile_inst.first << ".ptx" << "\n";
					//std::cerr << "Error opening file for writing: " << shader_to_compile_inst.first<<".ptx" << std::endl;
					return;
				}

				file << ptx.data();

				file.close();
			}

		}
		std::cout << compilationOutput.str() << std::endl;
		const char* skybox_path = "/Assets/Textures/autumn_field_puresky_2k.exr";
		Texture2D skybox = Texture2D::LoadImageFromFile(ProjectPath + "/Assets/Textures/autumn_field_puresky_2k.exr");
		// Texture2D skybox = Texture2D::LoadImageFromFile(ProjectPath + "/Assets/Textures/black.png");

		//构建dome light
		UniquePtrDevice DomeLightBuffer;
		UniquePtrDevice DomeLightBuffer_rowAlias;
		UniquePtrDevice DomeLightBuffer_rowQ;
		UniquePtrDevice DomeLightBuffer_colAlias;
		UniquePtrDevice DomeLightBuffer_colQ;
		UniquePtrDevice DomeLightBuffer_pdfMarginal;
		UniquePtrDevice DomeLightBuffer_pdfRow;
		{
			std::cout << "从" << skybox_path << "构建dome light" << endl;
			if (skybox.GetTextureView().textureFormat != TEXTURE_FORMAT_FLOAT4) {
		 		std::cerr << "请使用hdr环境贴图" << endl;
		 		system("pause");
			}
			uint w, h;
			float4* h_image = ReadOpenExr((ProjectPath+skybox_path).c_str(), w, h);
			// 拿到图片以后池化再取对数
			// 图片压缩为256x128的大小，取亮度的ln(x+1)作为概率并归一化
			// 最后得到两个buffer
			// 一个是行内的cdf，一个是行间的cdf

			// 首先申请照度的buffer
			UniquePtrDevice luminanceBuffer;
			CUDA_CHECK(cudaMalloc(luminanceBuffer.GetAddressOfPtr(), sizeof(float) * 128 * 256));
			
			DownSampleAndGetLuminance((float*)luminanceBuffer.GetPtr(), h_image, w, h);

			//ShowImage<float4>((float4*)h_image, w, h, false,"",false,false,false,false);
			// 1. 从 device 拷回未归一化 luminance
			std::vector<float> h_luminance = CopyFloatImageFromDevice((float*)luminanceBuffer.GetPtr(), DST_W, DST_H);

			// 2. CPU 构建 alias table
			EnvAliasTables tables = BuildEnvAliasTablesFromLuminance(h_luminance, DST_W, DST_H);

			// 申请5个buffer
			CUDA_CHECK(cudaMalloc(DomeLightBuffer.GetAddressOfPtr(), sizeof(DomeLightISStruct) ));
			CUDA_CHECK(cudaMalloc(DomeLightBuffer_rowQ.GetAddressOfPtr(), sizeof(float) * tables.width * tables.height));
			CUDA_CHECK(cudaMalloc(DomeLightBuffer_colQ.GetAddressOfPtr(), sizeof(float) * tables.height));
			CUDA_CHECK(cudaMalloc(DomeLightBuffer_colAlias.GetAddressOfPtr(), sizeof(int) * tables.height));
			CUDA_CHECK(cudaMalloc(DomeLightBuffer_rowAlias.GetAddressOfPtr(), sizeof(int) * tables.width * tables.height));
			// 装有pdf的两个buffer
			CUDA_CHECK(cudaMalloc(DomeLightBuffer_pdfRow.GetAddressOfPtr(), sizeof(float) * tables.width * tables.height));
			CUDA_CHECK(cudaMalloc(DomeLightBuffer_pdfMarginal.GetAddressOfPtr(), sizeof(float) * tables.height));
			CUDA_CHECK(cudaDeviceSynchronize());
			// 上传4个buffer
			CUDA_CHECK(cudaMemcpy(DomeLightBuffer_rowQ.GetPtr(), tables.rowQ.data(), sizeof(float) * tables.width * tables.height, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(DomeLightBuffer_colQ.GetPtr(), tables.colQ.data(), sizeof(float) * tables.height, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(DomeLightBuffer_colAlias.GetPtr(), tables.colAlias.data(), sizeof(int) * tables.height, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(DomeLightBuffer_rowAlias.GetPtr(), tables.rowAlias.data(), sizeof(int)* tables.width* tables.height, cudaMemcpyHostToDevice));

			CUDA_CHECK(cudaMemcpy(DomeLightBuffer_pdfRow.GetPtr(), tables.pdfRow.data(), sizeof(float)* tables.width* tables.height, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(DomeLightBuffer_pdfMarginal.GetPtr(), tables.pdfMarginal.data(), sizeof(float)* tables.height, cudaMemcpyHostToDevice));
			DomeLightISStruct tempStruct = {};
			tempStruct.width = tables.width;
			tempStruct.height = tables.height;
			tempStruct.colAlias = (int*)DomeLightBuffer_colAlias.GetPtr();
			tempStruct.rowAlias = (int*)DomeLightBuffer_rowAlias.GetPtr();
			tempStruct.colQ = (float*)DomeLightBuffer_colQ.GetPtr();
			tempStruct.rowQ = (float*)DomeLightBuffer_rowQ.GetPtr();
			tempStruct.pdfRow = (float*)DomeLightBuffer_pdfRow.GetPtr();
			tempStruct.pdfMarginal = (float*)DomeLightBuffer_pdfMarginal.GetPtr();
			// 上传装配好的buffer
			CUDA_CHECK(cudaMemcpy(DomeLightBuffer.GetPtr(), &tempStruct, sizeof(DomeLightISStruct), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		TextureManager::GetInstance().Add("skybox", skybox);
		uint KeyBoardActionBitMask = 0U;
		uint MouseActionBitMask = 0U;
		double2 MousePos;
		double2 MousePosPrev;
		//配置场景
		SceneManager scene;
		scene.WarmUp();
		RayTracingConfig conf;
		conf.NumSbtRecords = 1;
		conf.MaxRayRecursiveDepth = 8;
		conf.MaxSceneTraversalDepth = 2;
		conf.pipelineCompileOptions = CreatePipelineCompileOptions(OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY, 16, 2);
		scene.SetRayTracingConfig(conf);
		scene.ImportCompiledShader(ProjectPath + "/CompiledShaders/disney_principled_pt.cu.ptx", "module_disney_principled");
		// 加载一般的核函数
		string AccumulateCsPath = ProjectPath + "/CompiledShaders/kernels.cu.ptx";
		CUmodule AccumulateCs = LoadModule(AccumulateCsPath);
		CUfunction AccumulateCs_Fn = LoadFunction(AccumulateCs, "AccumulateFrame");

		// 申请相机调试buffer
		UniquePtrDevice DebugBuffer;
		UniquePtrDevice DebugBufferPayloadLength;
		UniquePtrDevice DepthBuffer;
		CUDA_CHECK(cudaMalloc(DebugBuffer.GetAddressOfPtr(), DEBUG_BUFF_SIZE));
		CUDA_CHECK(cudaMalloc(DebugBufferPayloadLength.GetAddressOfPtr(), sizeof(uint)));

		scene.AddMissShader("__miss__fetchMissInfo", "module_disney_principled");
		scene.AddRayGenerationShader("__raygen__principled_bsdf", "module_disney_principled");

		scene.AddHitShader("HitGroup_fetchHitInfo", "module_disney_principled", "__closesthit__fetch_hitinfo", "", "");

		scene.AddHitShader("HitGroup_fetchHitInfo_proceduralgeo_sphere_light", "module_disney_principled", "__closesthit__light", "", "__intersection__sphere_light");

		scene.AddHitShader("HitGroup_fetchHitInfo_proceduralgeo_rectangle_light", "module_disney_principled", "__closesthit__light", "", "__intersection__rectangle_light");
		{
			ObjLoadResult sponza = LoadObj(ProjectPath + "/Assets/scene1/optix_scene.obj");
			for (const auto& one : sponza) {
				const string& name = one.first;
				const Mesh& mesh = one.second.first;
				const Material& mat = one.second.second;
				ObjectDesc desc;
				desc.mesh = mesh;
				desc.mat = mat;
				desc.shaders = { "HitGroup_fetchHitInfo" };
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
				desc.shaders = { "HitGroup_fetchHitInfo" };
				scene.AddObjects(desc, name);
			}
		}
		
		{
			float3 corner = make_float3(0.2, 0.2, 1.3);
			RectangleLight rectangleLight1(make_float3(1, 1, 1)* corner,
				make_float3(1, -1, 1)* corner,
				make_float3(-1, 1, 1)* corner,
				make_float3(-1, -1, 1)* corner,
				make_float3(1, 1, 1), 12);
			string name = "rectangle_light1";
			scene.AddProceduralObject(
				name, rectangleLight1.GetAabb(),
				rectangleLight1.PackMaterialBuffer(),
				{ "HitGroup_fetchHitInfo_proceduralgeo_rectangle_light" }, true);
		}
		//{
		//	SphereLight SphereLight1(
		//		make_float3(-8.2, -2.76, 0.562), 0.15, make_float3(1, 0.3, 0.3), 200);
		//	string name = "sphere_light1";
		//	scene.AddProceduralObject(
		//		name, SphereLight1.GetAabb(),
		//		SphereLight1.PackMaterialBuffer(),
		//		{ "HitGroup_fetchHitInfo_proceduralgeo_sphere_light" }, true);
		//}
		const float R[7] = { 1.0, 1.0, 1.0, 0.05, 0.05, 0.05, 0.58 };  // 红色分量
		const float G[7] = { 0.05, 0.65, 1.0, 1.0, 1.0, 0.05, 0.05 };  // 绿色分量
		const float B[7] = { 0.05, 0.05, 0.05, 0.05, 1.0, 1.0, 0.83 };  // 蓝色分量
		for(uint rainbow_i=0; rainbow_i<7; rainbow_i++)
		{
			SphereLight SphereLight2(
				make_float3(-1+0.3* rainbow_i,-1,-1.16), 0.07, make_float3(R[rainbow_i], G[rainbow_i], B[rainbow_i]), 10);
			string name = "sphere_light_rainbow_"+std::to_string(rainbow_i);
			scene.AddProceduralObject(
				name, SphereLight2.GetAabb(),
				SphereLight2.PackMaterialBuffer(),
				{ "HitGroup_fetchHitInfo_proceduralgeo_sphere_light" }, true);
		}

		scene.ConfigureMissSbt({ make_float3(0,0,0),1.0f,skybox.GetTextureView() });
		scene.ConfigureRGSbt({ 1.0f,0.0f,1.0f });
		scene.BuildSceneWithProceduralGeometrySupported();
		LightManager::GetInstance().UploadLightList();
		MyCamera camera(make_float3(0,-5,0), make_float2(M_PI * 0.5, M_PI / 2));
		CameraData cameraData = camera.ExportCameraData(default_width, default_height);

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
		//创建渲染纹理用于帧结果累计
		//希望渲染纹理不要频繁申请释放
		const uint expected_max_width = (uint)GetSystemMetrics(SM_CXSCREEN);
		const uint expected_max_height = (uint)GetSystemMetrics(SM_CYSCREEN);
		UniquePtrDevice FrameBuffer2, FrameBuffer3;
		CUDA_CHECK(cudaMalloc(FrameBuffer2.GetAddressOfPtr(), sizeof(float3) * expected_max_width * expected_max_height));
		CUDA_CHECK(cudaMalloc(FrameBuffer3.GetAddressOfPtr(), sizeof(float3) * expected_max_width * expected_max_height));
		uint64 FrameNumber = 0;

		// 初始化随机数器
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

		auto initialTime = high_resolution_clock::now();

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

		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForOpenGL(window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
		ImGui_ImplOpenGL3_Init();
		disableMinimizeButton(window);

		io.FontGlobalScale = 1.5f;
		ConsoleOptions consoleOptions;
		ConsoleOptions* consoleOptionsDevice;
		CUDA_CHECK(cudaMalloc(&consoleOptionsDevice, sizeof(consoleOptionsDevice)));
		// 申请深度缓冲
		CUDA_CHECK(cudaMalloc(DepthBuffer.GetAddressOfPtr(), default_width * default_height * sizeof(float)));

		// 拷贝debugbuffer
		float3* points = (float3*)malloc(DEBUG_BUFF_SIZE);
		uint debug_buff_len;
		bool draw_light_path = false;

		std::cout << "\nHitObject调试颜色：\n粉色(Miss)\n绿色(灯光)\n黄色(不透明物体)\n红色(程序化几何体)" << std::endl;
		while (!glfwWindowShouldClose(window))
		{
			static int debug_mode_current_item = 0;
			static int accumulate_frame_mode=2;
			consoleOptions.debugMode =(ConsoleDebugMode)debug_mode_current_item;
			consoleOptions.frameAccumulationOptions = (FrameAccumulationOptions)accumulate_frame_mode;
			if (consoleOptions.debugMode != ConsoleDebugMode::NoDebug) {
				consoleOptions.frameAccumulationOptions = FrameAccumulationOptions::ForceOff;
				accumulate_frame_mode = 1;
			}
			
			CUDA_CHECK(cudaMemcpyAsync(consoleOptionsDevice, &consoleOptions, sizeof(ConsoleOptions), cudaMemcpyHostToDevice, Stream));

			int width, height;
			glfwGetWindowSize(window, &width, &height);
			glViewport(0, 0, width, height);
			if (prev_width != width || prev_height != height && accumulate_frame_mode!=0) {
				// 先拷贝下来
				LaunchParameters LParamsHost;
				CUDA_CHECK(cudaMemcpyAsync(&LParamsHost, LParams.GetPtr(), sizeof(LaunchParameters), cudaMemcpyDeviceToHost, Stream));
				CUDA_CHECK(cudaStreamSynchronize(Stream));
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
				LParamsHost.Width = width;
				LParamsHost.Height = height;
				//重新设置相机数据
				CameraData data = camera.ExportCameraData(width, height);
				LParamsHost.cameraData = data;
				//重新创建FrameBuffer2
				//重新累计
				FrameCounter = 0;
				if (width > expected_max_width || height > expected_max_height) {
					CUDA_CHECK(cudaFree(FrameBuffer2.GetPtr()));
					CUDA_CHECK(cudaMalloc(FrameBuffer2.GetAddressOfPtr(), sizeof(float3) * width * height));
					CUDA_CHECK(cudaFree(FrameBuffer3.GetPtr()));
					CUDA_CHECK(cudaMalloc(FrameBuffer3.GetAddressOfPtr(), sizeof(float3) * width * height));

					// 重新申请深度缓冲
					CUDA_CHECK(cudaFree(DepthBuffer.GetPtr()));
					CUDA_CHECK(cudaMalloc(DepthBuffer.GetAddressOfPtr(), width * height * sizeof(float)));
					//更新lparams中的参数
					LParamsHost.IndirectOutputBuffer= (float3*)FrameBuffer2.GetPtr();
					LParamsHost.DepthBuffer = (float*)DepthBuffer.GetPtr();
				}
				CUDA_CHECK(cudaDeviceSynchronize());
				CUDA_CHECK(cudaMemcpyAsync(LParams.GetPtr(), &LParamsHost, sizeof(LaunchParameters), cudaMemcpyHostToDevice, Stream));
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
				params.LightListArrayptr = LightManager::GetInstance().GetPtr();
				params.LightListLength = LightManager::GetInstance().GetLength();

				params.ImagePtr = devPtr;
				params.Width = width;
				params.Height = height;
				params.Handle = scene.GetTraversableHandle();
				params.Seed = static_cast<uint>(rand());
				params.cameraData = camera.ExportCameraData(width, height);
				params.Spp = 1;
				params.MaxRecursionDepth = scene.GetMaxRecursionDepth();
				params.IndirectOutputBuffer = (float3*)FrameBuffer2.GetPtr();
				params.consoleOptions = consoleOptionsDevice;
				params.DomeLightBuffer = DomeLightBuffer.GetPtr();
				params.DebugBuffer = DebugBuffer.GetPtr();
				params.DebugBufferPayloadLength = (uint*)DebugBufferPayloadLength.GetPtr();
				params.DepthBuffer = (float*)DepthBuffer.GetPtr();
				CUDA_CHECK(cudaMemcpyAsync(LParams.GetPtr(), &params, sizeof(LaunchParameters), cudaMemcpyHostToDevice, Stream));
			}

			//开始渲染逻辑
			//更新lparams
			{
				LaunchParameters LParamsHost;
				CUDA_CHECK(cudaMemcpyAsync(&LParamsHost, LParams.GetPtr(), sizeof(LaunchParameters), cudaMemcpyDeviceToHost, Stream));
				CUDA_CHECK(cudaStreamSynchronize(Stream));
				if (KeyBoardActionBitMask || MouseActionBitMask) {
					if (accumulate_frame_mode != 0)
						FrameCounter = 0;
					camera.Update(delta_time,
						KeyBoardActionBitMask,
						MouseActionBitMask,
						MousePos,
						MousePosPrev);
					//只拷贝相机数据
					CameraData data = camera.ExportCameraData(width, height);
					LParamsHost.cameraData = data;
					
				}
				//拷贝随机种
				uint seed = static_cast<uint>(rand());
				LParamsHost.Seed = seed;
				LParamsHost.FrameNumber = FrameNumber;
				CUDA_CHECK(cudaMemcpyAsync(LParams.GetPtr(), &LParamsHost, sizeof(LaunchParameters), cudaMemcpyHostToDevice, Stream));
				CUDA_CHECK(cudaStreamSynchronize(Stream));
				CUDA_CHECK(cudaGetLastError());
				scene.DispatchRays(devPtr, Stream, (LaunchParameters*)LParams.GetPtr(), width, height);
			}

			//希望当相机不动时累计结果
			{
				uint pixel_count = width * height;
				void* args[] = { &pixel_count,&FrameCounter,&devPtr,FrameBuffer2.GetAddressOfPtr(),FrameBuffer3.GetAddressOfPtr(),&consoleOptionsDevice};

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
				
				// 画调试线条
				if (debug_mode_current_item == 13 && draw_light_path) {
					CUDA_CHECK(cudaDeviceSynchronize());
					CameraData cdata = camera.ExportCameraData(width,height);
					bool first_time=true;
					bool nee=false;
					float* depth_buffer_ptr = (float*)DepthBuffer.GetPtr();
					for (int i = 0; i < debug_buff_len; ) {
						float line_width = 2.0f;
						if (first_time) {
							LaunchDrawWorldLineParallel(devPtr, depth_buffer_ptr, width, height, cdata.cam_eye, cdata.cam_u, cdata.cam_v, cdata.cam_w,
								points[i], points[i + 1], make_uchar4(0, 0, 255, 1), line_width, DepthTestWhenDebugLightPath);
						}
						else if (nee) {
							LaunchDrawWorldLineParallel(devPtr, depth_buffer_ptr, width, height, cdata.cam_eye, cdata.cam_u, cdata.cam_v, cdata.cam_w,
								points[i], points[i + 1], make_uchar4(0, 255, 0, 1), line_width, DepthTestWhenDebugLightPath);
						}
						else {
							LaunchDrawWorldLineParallel(devPtr, depth_buffer_ptr, width, height, cdata.cam_eye, cdata.cam_u, cdata.cam_v, cdata.cam_w,
								points[i], points[i + 1], make_uchar4(255, 0, 0, 1), line_width, DepthTestWhenDebugLightPath);
						}
						
						i = i + 2;
						if (first_time) {
							first_time = false;
						}
						else {
							nee = !nee;
						}
					}
					CUDA_CHECK(cudaDeviceSynchronize());
				}
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
			
			glfwPollEvents();

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::Begin("Debug Console");
			ImGui::Checkbox("DepthTestWhenDebugLightPath", &DepthTestWhenDebugLightPath);
			if (ImGui::Button("Capture"))
			{
				std::cout << "截取屏幕" << std::endl;
				uchar4* FrameBufferHost = (uchar4*)malloc(sizeof(uchar4) * width * height);
				CUDA_CHECK(cudaMemcpy(FrameBufferHost, devPtr, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
				SaveUchar4Image(FrameBufferHost, width, height, ("D:/OptixRenderer/ScreenShot/" + GetTimeString()+".png").c_str());
			}
			if (ImGui::Button("Capture LightPath") && debug_mode_current_item == 13) {
				CUDA_CHECK(cudaMemcpy(points, DebugBuffer.GetPtr(), DEBUG_BUFF_SIZE, cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(&debug_buff_len, DebugBufferPayloadLength.GetPtr(), sizeof(uint), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaDeviceSynchronize());
				for (int i = 0; i < debug_buff_len;) {
					std::cout << "起始：" << points[i].x << "," << points[i].y << "," << points[i].z;
					i += 1;
					std::cout << "  终点：" << points[i].x << "," << points[i].y << "," << points[i].z << std::endl;
					i += 1;
				}
				std::cout << "\n" << std::endl;
				draw_light_path = true;
			}
			ImGui::Combo("Debug Mode", &debug_mode_current_item, debugModeItems, IM_ARRAYSIZE(debugModeItems));
			ImGui::Combo("Frame Accumulation Mode", &accumulate_frame_mode, frameAccumulationItems, IM_ARRAYSIZE(frameAccumulationItems));
			ImGui::End();

			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			//更新游戏计数器
			auto end = high_resolution_clock::now();
			delta_time = duration_cast<milliseconds>(end - start).count();
			start = end;
			stringstream ss;
			auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - initialTime).count();
			ss << "Optix Renderer  fps:" << to_string_with_precision(1000.0f / delta_time,0) <<" W: "<< width << " H: "<<height << " Time elapsed:" << elapsed << " Spp:"<< FrameCounter << endl;
			glfwSetWindowTitle(window, ss.str().c_str());
			FrameNumber++;
			CUDA_CHECK(cudaStreamSynchronize(Stream));
			glfwSwapBuffers(window);
		}
		free(points);
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		CUDA_CHECK(cudaFree(consoleOptionsDevice));
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
		return;
	}
	catch (std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << "\n";
		system("pause");
	}
}