#pragma once

#include "common.h"
#include <utility>
#include <string>
#include <unordered_map>
#include <vector>

using std::vector;
using std::string;
using std::pair;

inline std::string readFileIntoString(const std::string& path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '" << path << "'" << std::endl;
        return "";
    }
    return { std::istreambuf_iterator<char>(input_file), std::istreambuf_iterator<char>() };
}

inline void ReadShaderHeaders(string ProjectDir, vector<pair<string, string>>& res) {
	string& ShaderLibraryDir = ProjectDir;
    for (const auto& entry : std::filesystem::directory_iterator(ShaderLibraryDir)) {
        if (entry.is_regular_file()) {  // 确保它是一个常规文件
            res.push_back({ entry.path().filename().string(), readFileIntoString(entry.path().string()) });
        }
    }
}

inline void ReadShaderHeadersRecursive(string ProjectDir, vector<pair<string, string>>& res) {
    string& ShaderLibraryDir = ProjectDir;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(ShaderLibraryDir)) {
        if (entry.is_regular_file()) {  // 确保它是一个常规文件
            res.push_back({ entry.path().filename().string(), readFileIntoString(entry.path().string()) });
        }
    }
}

typedef unordered_map<string, string> ShaderCollection;
inline ShaderCollection ReadShaderSources(string ProjectDir) {
    string ShaderDir = ProjectDir + "/Shaders";
    ShaderCollection collection;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(ShaderDir)) {
        if (entry.is_regular_file()) {  // 确保它是一个常规文件
            collection.insert({ entry.path().filename().string(),readFileIntoString(entry.path().string()) });
        }
    }
    return collection;
}
