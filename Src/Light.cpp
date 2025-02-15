#include "Light.h"

SphereLight::SphereLight(float3 Pos, float Radius, float3 Color)
{
    this->pos = Pos;
    this->radius = Radius;
    this->color = Color;
}

SphereLight::SphereLight(float3 Pos, float Radius, float3 Color, float intensity)
{
    this->pos = Pos;
    this->radius = Radius;
    this->color = make_float3(Color.x * intensity, Color.y * intensity, Color.z * intensity);
}
OptixAabb SphereLight::GetAabb()
{
    OptixAabb aabb;
    aabb.minX = this->pos.x - this->radius;
    aabb.minY = this->pos.y - this->radius;
    aabb.minZ = this->pos.z - this->radius;
    aabb.maxX = this->pos.x + this->radius;
    aabb.maxY = this->pos.y + this->radius;
    aabb.maxZ = this->pos.z + this->radius;
    return aabb;
}

// 元素0： 灯光序号
// 元素1： 灯光类型
// 元素2： color.x
// 元素3： color.y
// 元素4： color.z
// 元素5： pos.x
// 元素6： pos.y
// 元素7： pos.z
// 元素8： 半径
ProceduralGeometryMaterialBuffer SphereLight::PackMaterialBuffer()
{
    // 位置 3 float
    // 半径 1 float
    // 颜色 3 float
    ProceduralGeometryMaterialBuffer buffer;
    // 第0为用来放灯光序号
    uint* lightTypePtr = reinterpret_cast<uint*>(&buffer.Elements[1]);
    *lightTypePtr = LightType::Sphere;
    buffer.Elements[2] = this->color.x;
    buffer.Elements[3] = this->color.y;
    buffer.Elements[4] = this->color.z;
    buffer.Elements[5] = this->pos.x;
    buffer.Elements[6] = this->pos.y;
    buffer.Elements[7] = this->pos.z;
    buffer.Elements[8] = this->radius;
    
    return buffer;
}

float RectangleLight::max4(float a, float b, float c, float d)
{
    return fmaxf(a, fmaxf(b, fmaxf(c, d)));
}
float RectangleLight::min4(float a, float b, float c, float d)
{
    return fminf(a, fminf(b, fminf(c, d)));
}


RectangleLight::RectangleLight(float3 p1, float3 p2, float3 p3, float3 p4, float3 Color, float intensity)
{
    this->p1 = p1;
    this->p2 = p2;
    this->p3 = p3;
    this->p4 = p4;
    this->color = make_float3(Color.x * intensity, Color.y * intensity, Color.z * intensity);
}

OptixAabb RectangleLight::GetAabb()
{
    OptixAabb aabb;
    aabb.maxX = max4(p1.x, p2.x, p3.x, p4.x);
    aabb.maxY = max4(p1.y, p2.y, p3.y, p4.y);
    aabb.maxZ = max4(p1.z, p2.z, p3.z, p4.z);
    aabb.minX = min4(p1.x, p2.x, p3.x, p4.x);
    aabb.minY = min4(p1.y, p2.y, p3.y, p4.y);
    aabb.minZ = min4(p1.z, p2.z, p3.z, p4.z);

    aabb.maxX = fmaxf(aabb.minX + 1e-3f, aabb.maxX);
    aabb.maxY = fmaxf(aabb.minY + 1e-3f, aabb.maxY);
    aabb.maxZ = fmaxf(aabb.minZ + 1e-3f, aabb.maxZ);
    return aabb;
}

// 元素0： 灯光序号
// 元素1： 灯光类型
// 元素2： color.x
// 元素3： color.y
// 元素4： color.z
// 元素5： p1.x
// 元素6： p1.y
// 元素7： p1.z
// 元素8： p2.x
// 元素9： p2.y
// 元素10： p2.z
// 元素11： p3.x
// 元素12： p3.y
// 元素13： p3.z
// 只存储一个直角三角形，p1为直角的点
ProceduralGeometryMaterialBuffer RectangleLight::PackMaterialBuffer()
{
    ProceduralGeometryMaterialBuffer buffer;
    uint* lightTypePtr = reinterpret_cast<uint*>(&buffer.Elements[1]);
    *lightTypePtr = LightType::Area;
    buffer.Elements[2] = this->color.x;
    buffer.Elements[3] = this->color.y;
    buffer.Elements[4] = this->color.z;
    // 确定直角顶点
    float L12 = length(p1 - p2);
    float L13 = length(p1 - p3);
    float L23 = length(p2 - p3);
    if (L12 * L12 + L13 * L13 == L23 * L23) {
        // p1是直角顶点
        buffer.Elements[5] = p1.x;
        buffer.Elements[6] = p1.y;
        buffer.Elements[7] = p1.z;
        buffer.Elements[8] = p2.x;
        buffer.Elements[9] = p2.y;
        buffer.Elements[10] = p2.z;
        buffer.Elements[11] = p3.x;
        buffer.Elements[12] = p3.y;
        buffer.Elements[13] = p3.z;
    }
    else if (L12 * L12 + L23 * L23 == L13 * L13) {
        // p2是直角顶点
        buffer.Elements[5] = p2.x;
        buffer.Elements[6] = p2.y;
        buffer.Elements[7] = p2.z;
        buffer.Elements[8] = p1.x;
        buffer.Elements[9] = p1.y;
        buffer.Elements[10] = p1.z;
        buffer.Elements[11] = p3.x;
        buffer.Elements[12] = p3.y;
        buffer.Elements[13] = p3.z;
    }
    else if (L13 * L13 + L23 * L23 == L12* L12) {
        // p3是直角顶点
        buffer.Elements[5] = p3.x;
        buffer.Elements[6] = p3.y;
        buffer.Elements[7] = p3.z;
        buffer.Elements[8] = p1.x;
        buffer.Elements[9] = p1.y;
        buffer.Elements[10] = p1.z;
        buffer.Elements[11] = p2.x;
        buffer.Elements[12] = p2.y;
        buffer.Elements[13] = p2.z;
    }
    else {
        throw std::runtime_error("not a rectangle");
    }
    return buffer;
}
