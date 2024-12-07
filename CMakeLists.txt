cmake_minimum_required(VERSION 3.20)
project(OptixRenderer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};E:/vcpkg/vcpkg/packages/libdeflate_x64-windows)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};E:/vcpkg/vcpkg/packages/glfw3_x64-windows)
file(GLOB CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/*.cu)
file(GLOB CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Src/*.cpp)

find_package(CUDA REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenEXR CONFIG REQUIRED)

set(OPTIX_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")

add_executable(Main ${CPP_SOURCES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Header)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty)
include_directories(${OPTIX_PATH}/include)
include_directories(${OPTIX_PATH}/SDK/support)
include_directories(${OPTIX_PATH}/SDK/sutil)
include_directories(${OPTIX_PATH}/SDK)
include_directories(${OPTIX_PATH}/SDK/support/glad)
include_directories(${OPTIX_PATH}/SDK/support/GLFW/include/GLFW)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ShaderLibrary)

target_link_libraries(Main PRIVATE cudart)
target_link_libraries(Main PRIVATE curand)
target_link_libraries(Main PRIVATE nvrtc)
target_link_libraries(Main PRIVATE cuda)
target_link_libraries(Main PRIVATE OpenEXR::OpenEXR)
target_link_libraries(Main PRIVATE glfw)
enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# set(INCLUDE_DIRS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/Header)
# set(INCLUDE_DIRS_NATIVE_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/ShaderLibrary)
# set(INCLUDE_DIRS_OPTIX ${OPTIX_PATH}/include)
# set(INCLUDE_DIRS_OPTIX_SDK ${OPTIX_PATH}/SDK)
# set(INCLUDE_DIRS_CUDA ${CUDA_PATH}/include)
# foreach(CU_FILE ${CUDA_SOURCES})
#     get_filename_component(FILE_NAME ${CU_FILE} NAME_WE)
#     set(PTX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/CompiledShaders/${FILE_NAME}.ptx)

#     set_source_files_properties(${CU_FILE} PROPERTIES CUDA_PTX_COMPILATION ON)
#     add_custom_command(
#         OUTPUT ${PTX_FILE}
#         COMMAND ${CMAKE_CUDA_COMPILER} -ptx ${CU_FILE} -o ${PTX_FILE} -I ${INCLUDE_DIRS_NATIVE_CUDA} -I ${INCLUDE_DIRS_OPTIX} -I ${INCLUDE_DIRS_CUDA} -I ${INCLUDE_DIRS_OPTIX_SDK} -I ${INCLUDE_DIRS_CPP}
#         DEPENDS ${CU_FILE}
#         COMMENT "Compiling ${CU_FILE} to PTX"
#     )
#     list(APPEND PTX_FILES ${PTX_FILE})
# endforeach()

# add_custom_target(
#     GeneratePTX ALL
#     DEPENDS ${PTX_FILES}
# )
