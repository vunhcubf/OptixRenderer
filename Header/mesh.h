#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <optix.h>
#include <fstream>
#include <vector_types.h>
#include <vector_functions.h>

#include "common.h"
struct MyMesh {
	static MyMesh LoadMeshFromFile(std::string path);
	~MyMesh();
	size_t GetVerticesCount();
	size_t GetIndicesCount();
	void* GetVerticesPtr();
	void* GetNormalsPtr();
	void* GetUVsPtr();
	void* GetIndicesPtr();

	std::vector<float3> Vertices;
	std::vector<float3> Normals;
	std::vector<float2> Uvs;
	std::vector<uint3> Indices;
};