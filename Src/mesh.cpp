#include "mesh.h"

MyMesh MyMesh::LoadMeshFromFile(std::string path)
{
	std::vector<float3> vertices_temp;
	std::vector<float3> normal_temp;
	std::vector<float2> uv_temp;
	std::vector<float3> vertices;
	std::vector<float3> normal;
	std::vector<float2> uv;
	std::vector<uint3> indicesbuffer;
	uint vertex_counter=0;

	std::ifstream ifs;
	ifs.open(path);
	if (!ifs.is_open()) {
		std::stringstream ss;
		ss << "文件没有打开：" << path;
		throw std::exception(ss.str().c_str());
	}
	if (ifs.is_open()) {
		std::string buffer;
		while (std::getline(ifs, buffer)) {
			//是否为顶点坐标
			if (buffer[0] == 'v') {
				if (buffer[1] == ' ') {
					std::istringstream iss(buffer.substr(2));
					float x, y, z;
					iss >> x >> y >> z;
					vertices_temp.push_back(make_float3(x, y, z));
				}
				//是否为纹理坐标
				else if (buffer[1] == 't' && buffer[2] == ' ') {
					std::istringstream iss(buffer.substr(3));
					float x, y;
					iss >> x >> y;
					uv_temp.push_back(make_float2(x, y));
				}
				//是否为法线
				else if (buffer[1] == 'n' && buffer[2] == ' ') {
					std::istringstream iss(buffer.substr(3));
					float x, y, z;
					iss >> x >> y >> z;
					normal_temp.push_back(make_float3(x, y, z));
				}
			}
			//索引信息
			else if (buffer[0] == 'f' && buffer[1] == ' ') {
				std::string s = buffer.substr(2);
				bool is_v1_vn1 = false;
				int slope_counter = 0;
				for (auto it = s.begin(); it != s.end(); it++) {
					if (*it == '/') {
						slope_counter++;
						*it = ' ';
						if (*(it + 1) == '/' && !is_v1_vn1) {
							is_v1_vn1 = true;
						}
					}
				}
				if (is_v1_vn1) {
					uint v1, v2, v3, vn1, vn2, vn3;
					std::istringstream iss(s);
					iss >> v1 >> vn1 >> v2 >> vn2 >> v3 >> vn3;
					indicesbuffer.push_back(make_uint3(vertex_counter, vertex_counter+1, vertex_counter+2));
					vertices.push_back(vertices_temp[v1 - 1]);
					vertices.push_back(vertices_temp[v2 - 1]);
					vertices.push_back(vertices_temp[v3 - 1]);
					normal.push_back(normal_temp[vn1 - 1]);
					normal.push_back(normal_temp[vn2 - 1]);
					normal.push_back(normal_temp[vn3 - 1]);
					vertex_counter += 3;
				}
				else if (slope_counter == 6) {
					uint v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3;
					std::istringstream iss(s);
					iss >> v1 >> vt1 >> vn1 >> v2 >> vt2 >> vn2 >> v3 >> vt3 >> vn3;
					indicesbuffer.push_back(make_uint3(vertex_counter, vertex_counter + 1, vertex_counter + 2));
					vertices.push_back(vertices_temp[v1 - 1]);
					vertices.push_back(vertices_temp[v2 - 1]);
					vertices.push_back(vertices_temp[v3 - 1]);
					normal.push_back(normal_temp[vn1 - 1]);
					normal.push_back(normal_temp[vn2 - 1]);
					normal.push_back(normal_temp[vn3 - 1]);
					uv.push_back(uv_temp[vt1 - 1]);
					uv.push_back(uv_temp[vt2 - 1]);
					uv.push_back(uv_temp[vt3 - 1]);
					vertex_counter += 3;
				}
				else if (slope_counter == 3) {
					uint v1, vt1, v2, vt2, v3, vt3;
					std::istringstream iss(s);
					iss >> v1 >> vt1 >> v2 >> vt2 >> v3 >> vt3;
					indicesbuffer.push_back(make_uint3(vertex_counter, vertex_counter + 1, vertex_counter + 2));
					vertices.push_back(vertices_temp[v1 - 1]);
					vertices.push_back(vertices_temp[v2 - 1]);
					vertices.push_back(vertices_temp[v3 - 1]);
					uv.push_back(uv_temp[vt1 - 1]);
					uv.push_back(uv_temp[vt2 - 1]);
					uv.push_back(uv_temp[vt3 - 1]);
					vertex_counter += 3;
				}
			}
		}
	}
	ifs.close();
	MyMesh output;
	output.Indices = indicesbuffer;
	output.Normals = normal;
	output.Vertices = vertices;
	output.Uvs = uv;
	if (vertices.size() < 3) {
		std::stringstream ss;
		ss << "模型至少一个三角形，要么是三角形太少，要么是出错了 尝试读取的文件是：" << path;
		throw std::exception(ss.str().c_str());
	}
	return output;
}

MyMesh::~MyMesh()
{
	Vertices.clear();
	Vertices.shrink_to_fit();
	Normals.clear();
	Normals.shrink_to_fit();
	Uvs.clear();
	Uvs.shrink_to_fit();
	Indices.clear();
	Indices.shrink_to_fit();
}

size_t MyMesh::GetVerticesCount()
{
	return Vertices.size();
}

size_t MyMesh::GetIndicesCount()
{
	return Indices.size();
}

void* MyMesh::GetVerticesPtr()
{
	return Vertices.data();
}

void* MyMesh::GetNormalsPtr()
{
	return Normals.data();
}

void* MyMesh::GetUVsPtr()
{
	return Uvs.data();
}

void* MyMesh::GetIndicesPtr()
{
	return Indices.data();
}
