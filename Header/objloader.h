#pragma once
#include <filesystem>
#include "common.h"
#include "mesh.h"
#include <string>
#include <sstream>
#include <vector>
#include <exception>
#include <algorithm>
#include "Texture.h"
#include <bitset>
using std::vector;
using std::string;
typedef std::unordered_map<string, std::pair<Mesh, Material>> ObjLoadResult;

template<typename T>
inline T SafeAccessVector(vector<T>& t,uint i,uint line_number,const char* file_name) {
#ifdef _DEBUG
	try {
		return t.at(i);
	}
	catch (auto& e) {
		stringstream ss;
		ss << "��Ч���±���ʣ��ļ���" << file_name << " �кţ�" << line_number;
		throw std::runtime_error(ss.str());
	}
#else
	return t.at(i);
#endif
}
#define At(t,i) SafeAccessVector(t,i, __FILE__, __LINE__)

inline vector<string> Split(const string& text, char delim) {
	std::vector<std::string> tokens;
	size_t start = 0;
	size_t end = text.find(delim);

	while (end != std::string::npos) {
		if (end != start) {
			tokens.push_back(text.substr(start, end - start));
		}
		start = end + 1;
		end = text.find(delim, start);
	}
	if (start != text.length()) {
		tokens.push_back(text.substr(start));
	}

	return tokens;
}

inline unordered_map<string, Material> LoadMtl(string path) {
	unordered_map<string, Material> mats;
	vector<Material> materials;
	vector<string> names;
	std::ifstream ifs;
	ifs.open(path);
	if (!ifs.is_open()) {
		std::stringstream ss;
		ss << "�ļ�û�д򿪣�" << path;
		throw std::exception(ss.str().c_str());
	}
	if (ifs.is_open()) {
		string line;
		bool FirstTime = true;
		Material Temp;
		while (std::getline(ifs, line)) {
			vector<string> tokens = Split(line, ' ');
			if (tokens.size() == 0) {
				continue;
			}
			if (tokens.at(0) == "newmtl") {
				// �µĲ���
				names.push_back(tokens.at(1));
				if (FirstTime) {
					FirstTime = false;
				}
				else {
					materials.push_back(Temp);
					ResetMaterial(Temp);
				}
			}
			else if (tokens.at(0) == "Kd") {
				Temp.BaseColor = make_float3(std::atof(tokens.at(1).c_str()), std::atof(tokens.at(2).c_str()), std::atof(tokens.at(3).c_str()));
			}
			else if (tokens.at(0) == "Ks") {
				Temp.Specular = (std::atof(tokens.at(1).c_str()) + std::atof(tokens.at(2).c_str()) + std::atof(tokens.at(3).c_str())) / 3.0f;
			}
			else if (tokens.at(0) == "Ke") {
				Temp.Emission = make_float3(std::atof(tokens.at(1).c_str()), std::atof(tokens.at(2).c_str()), std::atof(tokens.at(3).c_str()));
			}
			else if (tokens.at(0) == "Ni") {
				Temp.Ior = std::atof(tokens.at(1).c_str());
			}
			else if (tokens.at(0) == "d") {
				Temp.Opacity = std::atof(tokens.at(1).c_str());
			}
			else if (tokens.at(0) == "Tf") {
				Temp.Transmission = (std::atof(tokens.at(1).c_str()) + std::atof(tokens.at(2).c_str()) + std::atof(tokens.at(3).c_str())) / 3.0f;
			}
			else if (tokens.at(0) == "Pm") {
				Temp.Metallic = std::atof(tokens.at(1).c_str());
			}
			else if (tokens.at(0) == "Pr") {
				Temp.Roughness = std::atof(tokens.at(1).c_str());
			}
		}
		Temp.MaterialType=MaterialType::MATERIAL_OBJ;
		materials.push_back(Temp);
	}
	for (uint i = 0; i < names.size(); i++) {
		mats.insert({ names.at(i),materials.at(i) });
	}
	return mats;
}

// Ԫ���е�string��ʾ���������
inline unordered_map<string, Material> LoadMtl(string path, unordered_map<string,tuple<string,Texture2D>>& TextureResources) {
	unordered_map<string, Material> mats;
	vector<Material> materials;
	vector<string> names;
	std::ifstream ifs;
	unordered_map<string, Texture2D> textureCollectionDiffuse;
	unordered_map<string, Texture2D> textureCollectionNormal;
	unordered_map<string, Texture2D> textureCollectionARM;
	ifs.open(path);
	if (!ifs.is_open()) {
		std::stringstream ss;
		ss << "�ļ�û�д򿪣�" << path;
		throw std::exception(ss.str().c_str());
	}
	if (ifs.is_open()) {
		string line;
		bool FirstTime = true;
		Material Temp;
		while (std::getline(ifs, line)) {
			vector<string> tokens = Split(line, ' ');
			if (tokens.size() == 0) {
				continue;
			}
			if (tokens.at(0) == "newmtl") {
				// �µĲ���
				names.push_back(tokens.at(1));
				if (FirstTime) {
					FirstTime = false;
				}
				else {
					materials.push_back(Temp);
					ResetMaterial(Temp);
				}
			}
			else if (tokens.at(0) == "Kd") {
				Temp.BaseColor = make_float3(std::atof(tokens.at(1).c_str()), std::atof(tokens.at(2).c_str()), std::atof(tokens.at(3).c_str()));
			}
			else if (tokens.at(0) == "Ks") {
				Temp.Specular = (std::atof(tokens.at(1).c_str()) + std::atof(tokens.at(2).c_str()) + std::atof(tokens.at(3).c_str())) / 3.0f;
			}
			else if (tokens.at(0) == "Ke") {
				Temp.Emission = make_float3(std::atof(tokens.at(1).c_str()), std::atof(tokens.at(2).c_str()), std::atof(tokens.at(3).c_str()));
			}
			else if (tokens.at(0) == "Ni") {
				Temp.Ior = std::atof(tokens.at(1).c_str());
			}
			else if (tokens.at(0) == "d") {
				Temp.Opacity = std::atof(tokens.at(1).c_str());
			}
			else if (tokens.at(0) == "Tf") {
				Temp.Transmission = (std::atof(tokens.at(1).c_str()) + std::atof(tokens.at(2).c_str()) + std::atof(tokens.at(3).c_str())) / 3.0f;
			}
			else if (tokens.at(0) == "Pm") {
				Temp.Metallic = std::atof(tokens.at(1).c_str());
			}
			else if (tokens.at(0) == "Pr") {
				Temp.Roughness = std::atof(tokens.at(1).c_str());
			}
			else if(tokens.at(0) == "map_Kd"){
				auto tex=Texture2D::LoadImageFromFile(tokens.at(1));
				tex.SetIfReleaseGpuArrayWhenDispose(false);
				textureCollectionDiffuse[names.back()] = tex;
			}
			else if(tokens.at(0) == "map_Ns"){
				auto tex=Texture2D::LoadImageFromFile(tokens.at(1));
				tex.SetIfReleaseGpuArrayWhenDispose(false);
				textureCollectionNormal[names.back()] = tex;
			}
			else if(tokens.at(0) == "map_Arm"){
				auto tex=Texture2D::LoadImageFromFile(tokens.at(1));
				tex.SetIfReleaseGpuArrayWhenDispose(false);
				textureCollectionARM[names.back()] = tex;
			}
		}
		Temp.MaterialType=MaterialType::MATERIAL_OBJ;
		materials.push_back(Temp);
	}
	for (uint i = 0; i < names.size(); i++) {
		if (textureCollectionDiffuse.find(names.at(i)) != textureCollectionDiffuse.end()) {
			materials.at(i).BaseColorMap = textureCollectionDiffuse[names.at(i)].GetTextureView();
			TextureResources[names.at(i)] = { "BaseColorMap",textureCollectionDiffuse[names.at(i)] };
		}
		else{
			materials.at(i).BaseColorMap = {0,0,0,0};
		}

		if (textureCollectionNormal.find(names.at(i)) != textureCollectionNormal.end()) {
			materials.at(i).NormalMap = textureCollectionNormal[names.at(i)].GetTextureView();
			TextureResources[names.at(i)] = { "NormalMap",textureCollectionNormal[names.at(i)] };
		}
		else{
			materials.at(i).NormalMap = {0,0,0,0};
		}

		if (textureCollectionARM.find(names.at(i)) != textureCollectionARM.end()) {
			materials.at(i).ARMMap = textureCollectionARM[names.at(i)].GetTextureView();
			TextureResources[names.at(i)] = { "ARMMap",textureCollectionARM[names.at(i)] };
		}
		else{
			materials.at(i).ARMMap = {0,0,0,0};
		}
		mats.insert({ names.at(i),materials.at(i) });
	}
	return mats;
}


inline ObjLoadResult LoadObj(string path) {
	// ���ɻ����������ظ�����
	vector<float3> vertices_temp;
	vector<float3> normal_temp;
	vector<float2> uv_temp;

	vector<float3> vertices;
	vector<float3> normal;
	vector<float2> uv;
	vector<uint3> indicesbuffer;
	uint vertex_counter = 0;
	unordered_map<string, Material> Mats;

	string MtlFileName = "";
	std::filesystem::path filePath = path;
	string dirPath = filePath.parent_path().string();

	ObjLoadResult res;
	vector<Mesh> Meshes;
	vector<string> Names;
	vector<Material> mat_temp;
	Material CurrentMatUsed;
	std::ifstream ifs;
	ifs.open(path);
	if (!ifs.is_open()) {
		std::stringstream ss;
		ss << "�ļ�û�д򿪣�" << path;
		throw std::exception(ss.str().c_str());
	}
	if (ifs.is_open()) {
		string line;
		// ��¼ģ�����з��ߡ�uv�����Ƕ���
		bool HasNormal = false;
		bool HasUv = false;
		while (std::getline(ifs, line)) {
			// ���Ƚ�һ�з�Ϊ����С��
			
			vector<string> tokens = Split(line, ' ');
			// ��ʼ���� ֻ֧�ֽ��� v vt vn f mtllib # o usemtl s 
			if (tokens.at(0) == "v") {
				// ��������
				vertices_temp.push_back(make_float3(
					std::atof(tokens.at(1).c_str()),
					std::atof(tokens.at(2).c_str()),
					std::atof(tokens.at(3).c_str())
					));
			}
			else if (tokens.at(0) == "vt") {
				// ��������
				HasUv = true;
				uv_temp.push_back(make_float2(
					std::atof(tokens.at(1).c_str()),
					std::atof(tokens.at(2).c_str())
				));
			}
			else if (tokens.at(0) == "vn") {
				// ����
				HasNormal = true;
				normal_temp.push_back(make_float3(
					std::atof(tokens.at(1).c_str()),
					std::atof(tokens.at(2).c_str()),
					std::atof(tokens.at(3).c_str())
				));
			}
			else if (tokens.at(0) == "f") {
				// ����
				// ���ģ��û�з��ߺ��������꣬��֧��
				if (!(HasUv && HasNormal)) {
					throw std::runtime_error("��Ҫģ��ͬʱ���з��ߺ���������");
				}
				if (tokens.size() == 5) {
					throw std::runtime_error("��֧���ı���");
				}
				uint v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3;
				std::replace(tokens.at(1).begin(), tokens.at(1).end(), '/', ' ');
				std::istringstream iss1(tokens.at(1));
				iss1 >> v1 >> vt1 >> vn1;

				std::replace(tokens.at(2).begin(), tokens.at(2).end(), '/', ' ');
				std::istringstream iss2(tokens.at(2));
				iss2 >> v2 >> vt2 >> vn2;

				std::replace(tokens.at(3).begin(), tokens.at(3).end(), '/', ' ');
				std::istringstream iss3(tokens.at(3));
				iss3 >> v3 >> vt3 >> vn3;

				indicesbuffer.push_back(make_uint3(vertex_counter, vertex_counter + 1, vertex_counter + 2));
				vertices.push_back(vertices_temp.at(v1 - 1));
				vertices.push_back(vertices_temp.at(v2 - 1));
				vertices.push_back(vertices_temp.at(v3 - 1));
				normal.push_back(normal_temp.at(vn1 - 1));
				normal.push_back(normal_temp.at(vn2 - 1));
				normal.push_back(normal_temp.at(vn3 - 1));
				uv.push_back(uv_temp.at(vt1 - 1));
				uv.push_back(uv_temp.at(vt2 - 1));
				uv.push_back(uv_temp.at(vt3 - 1));
				vertex_counter += 3;
			}
			else if (tokens.at(0) == "mtllib") {
				// �����ļ�
				MtlFileName = tokens.at(1);
				Mats = LoadMtl(dirPath +"/"+ MtlFileName);
			}
			else if (tokens.at(0) == "o") {
				// ��������
				// ��ʾ��ʼ����һ���µ����壬��Ҫ�����µ�������
				Names.push_back(tokens.at(1));
				if (Names.size() == 1) {
					// ��һ������
				}
				else {
					Mesh one;
					one.Indices = indicesbuffer;
					one.Normals = normal;
					one.Vertices = vertices;
					one.Uvs = uv;
					Meshes.push_back(one);
					mat_temp.push_back(CurrentMatUsed);
				}
				vertices.clear();
				normal.clear();
				uv.clear();
				indicesbuffer.clear();
			}
			else if (tokens.at(0) == "usemtl") {
				// ��������
				ResetMaterial(CurrentMatUsed);
				if (tokens.size() != 1) {
					auto res = Mats.find(tokens.at(1));
					if (res != Mats.end()) {
						CurrentMatUsed = res->second;
					}
				}
			}
			else if (tokens.at(0) == "#" || tokens.at(0) == "s") {
				// ����Ҫ�Ķ���
			}
			else {
				throw std::runtime_error("��֧�ֵĹؼ���");
			}
		}
		// ���һ������û����ʽ��o����β
		Mesh one;
		one.Indices = indicesbuffer;
		one.Normals = normal;
		one.Vertices = vertices;
		one.Uvs = uv;
		Meshes.push_back(one);
		mat_temp.push_back(CurrentMatUsed);
		for (uint i = 0; i < Names.size(); i++) {
			res.insert({ Names.at(i),{Meshes.at(i),mat_temp.at(i)} });
		}

		return res;
	}
}

inline ObjLoadResult LoadObj(string path, unordered_map<string, tuple<string, Texture2D>>& TextureResources) {
	// ���ɻ����������ظ�����
	vector<float3> vertices_temp;
	vector<float3> normal_temp;
	vector<float2> uv_temp;

	vector<float3> vertices;
	vector<float3> normal;
	vector<float2> uv;
	vector<uint3> indicesbuffer;
	uint vertex_counter = 0;
	unordered_map<string, Material> Mats;

	string MtlFileName = "";
	std::filesystem::path filePath = path;
	string dirPath = filePath.parent_path().string();

	ObjLoadResult res;
	vector<Mesh> Meshes;
	vector<string> Names;
	vector<Material> mat_temp;
	Material CurrentMatUsed;
	std::ifstream ifs;
	ifs.open(path);
	if (!ifs.is_open()) {
		std::stringstream ss;
		ss << "�ļ�û�д򿪣�" << path;
		throw std::exception(ss.str().c_str());
	}
	if (ifs.is_open()) {
		string line;
		// ��¼ģ�����з��ߡ�uv�����Ƕ���
		bool HasNormal = false;
		bool HasUv = false;
		while (std::getline(ifs, line)) {
			// ���Ƚ�һ�з�Ϊ����С��
			
			vector<string> tokens = Split(line, ' ');
			// ��ʼ���� ֻ֧�ֽ��� v vt vn f mtllib # o usemtl s 
			if (tokens.at(0) == "v") {
				// ��������
				vertices_temp.push_back(make_float3(
					std::atof(tokens.at(1).c_str()),
					std::atof(tokens.at(2).c_str()),
					std::atof(tokens.at(3).c_str())
					));
			}
			else if (tokens.at(0) == "vt") {
				// ��������
				HasUv = true;
				uv_temp.push_back(make_float2(
					std::atof(tokens.at(1).c_str()),
					std::atof(tokens.at(2).c_str())
				));
			}
			else if (tokens.at(0) == "vn") {
				// ����
				HasNormal = true;
				normal_temp.push_back(make_float3(
					std::atof(tokens.at(1).c_str()),
					std::atof(tokens.at(2).c_str()),
					std::atof(tokens.at(3).c_str())
				));
			}
			else if (tokens.at(0) == "f") {
				// ����
				// ���ģ��û�з��ߺ��������꣬��֧��
				if (!(HasUv && HasNormal)) {
					throw std::runtime_error("��Ҫģ��ͬʱ���з��ߺ���������");
				}
				if (tokens.size() == 5) {
					throw std::runtime_error("��֧���ı���");
				}
				uint v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3;
				std::replace(tokens.at(1).begin(), tokens.at(1).end(), '/', ' ');
				std::istringstream iss1(tokens.at(1));
				iss1 >> v1 >> vt1 >> vn1;

				std::replace(tokens.at(2).begin(), tokens.at(2).end(), '/', ' ');
				std::istringstream iss2(tokens.at(2));
				iss2 >> v2 >> vt2 >> vn2;

				std::replace(tokens.at(3).begin(), tokens.at(3).end(), '/', ' ');
				std::istringstream iss3(tokens.at(3));
				iss3 >> v3 >> vt3 >> vn3;

				indicesbuffer.push_back(make_uint3(vertex_counter, vertex_counter + 1, vertex_counter + 2));
				vertices.push_back(vertices_temp.at(v1 - 1));
				vertices.push_back(vertices_temp.at(v2 - 1));
				vertices.push_back(vertices_temp.at(v3 - 1));
				normal.push_back(normal_temp.at(vn1 - 1));
				normal.push_back(normal_temp.at(vn2 - 1));
				normal.push_back(normal_temp.at(vn3 - 1));
				uv.push_back(uv_temp.at(vt1 - 1));
				uv.push_back(uv_temp.at(vt2 - 1));
				uv.push_back(uv_temp.at(vt3 - 1));
				vertex_counter += 3;
			}
			else if (tokens.at(0) == "mtllib") {
				// �����ļ�
				MtlFileName = tokens.at(1);
				Mats = LoadMtl(dirPath +"/"+ MtlFileName,TextureResources);
			}
			else if (tokens.at(0) == "o") {
				// ��������
				// ��ʾ��ʼ����һ���µ����壬��Ҫ�����µ�������
				Names.push_back(tokens.at(1));
				if (Names.size() == 1) {
					// ��һ������
				}
				else {
					Mesh one;
					one.Indices = indicesbuffer;
					one.Normals = normal;
					one.Vertices = vertices;
					one.Uvs = uv;
					Meshes.push_back(one);
					mat_temp.push_back(CurrentMatUsed);
				}
				vertices.clear();
				normal.clear();
				uv.clear();
				indicesbuffer.clear();
			}
			else if (tokens.at(0) == "usemtl") {
				// ��������
				ResetMaterial(CurrentMatUsed);
				if (tokens.size() != 1) {
					auto res = Mats.find(tokens.at(1));
					if (res != Mats.end()) {
						CurrentMatUsed = res->second;
					}
				}
			}
			else if (tokens.at(0) == "#" || tokens.at(0) == "s") {
				// ����Ҫ�Ķ���
			}
			else {
				throw std::runtime_error("��֧�ֵĹؼ���");
			}
		}
		// ���һ������û����ʽ��o����β
		Mesh one;
		one.Indices = indicesbuffer;
		one.Normals = normal;
		one.Vertices = vertices;
		one.Uvs = uv;
		Meshes.push_back(one);
		mat_temp.push_back(CurrentMatUsed);
		for (uint i = 0; i < Names.size(); i++) {
			res.insert({ Names.at(i),{Meshes.at(i),mat_temp.at(i)} });
		}

		return res;
	}
}