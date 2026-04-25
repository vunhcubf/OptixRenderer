// ============================================================
// Alias Table
// ============================================================
#pragma once
#include "common.h"
#include <iostream>
#include <sstream>

typedef unsigned int uint;


#ifndef DST_W
#define DST_W 256
#endif

#ifndef DST_H
#define DST_H 128
#endif
struct AliasTable
{
    std::vector<float> q;
    std::vector<int> alias;
};

struct EnvAliasTables
{
    int width = 0;
    int height = 0;

    std::vector<float> rowQ;      // height * width
    std::vector<int>   rowAlias;  // height * width

    std::vector<float> colQ;      // height
    std::vector<int>   colAlias;  // height

    std::vector<float> pdfMarginal;
    std::vector<float> pdfRow;
};

struct DomeLightISStruct {
    int width = 0;
    int height = 0;

    float* rowQ;      // height * width
    int*   rowAlias;  // height * width

    float* colQ;      // height
    int*   colAlias;  // height

    float* pdfMarginal;
    float* pdfRow;
};

struct SampledPoint
{
    int row;
    int col;
};

inline AliasTable BuildAliasTable1D(const float* weights, int n)
{
    AliasTable table;
    table.q.resize(n);
    table.alias.resize(n);

    std::vector<float> prob(n, 0.0f);
    std::vector<float> scaled(n, 0.0f);
    std::vector<int> sMALL;
    std::vector<int> large;
    sMALL.reserve(n);
    large.reserve(n);

    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        float w = weights[i];
        if (!std::isfinite(w) || w < 0.0f)
            w = 0.0f;
        prob[i] = w;
        sum += w;
    }

    if (sum > 0.0f && std::isfinite(sum))
    {
        for (int i = 0; i < n; ++i)
            scaled[i] = prob[i] / sum * float(n);
    }
    else
    {
        for (int i = 0; i < n; ++i)
            scaled[i] = 1.0f;
    }

    for (int i = 0; i < n; ++i)
    {
        if (scaled[i] < 1.0f)
            sMALL.push_back(i);
        else
            large.push_back(i);
    }

    while (!sMALL.empty() && !large.empty())
    {
        int s = sMALL.back();
        sMALL.pop_back();

        int l = large.back();
        large.pop_back();

        table.q[s] = scaled[s];
        table.alias[s] = l;

        scaled[l] = scaled[l] - (1.0f - scaled[s]);

        if (scaled[l] < 1.0f)
            sMALL.push_back(l);
        else
            large.push_back(l);
    }

    while (!large.empty())
    {
        int l = large.back();
        large.pop_back();
        table.q[l] = 1.0f;
        table.alias[l] = l;
    }

    while (!sMALL.empty())
    {
        int s = sMALL.back();
        sMALL.pop_back();
        table.q[s] = 1.0f;
        table.alias[s] = s;
    }

    return table;
}

inline EnvAliasTables BuildEnvAliasTablesFromLuminance(
    std::vector<float>& luminance,
    int width,
    int height)
{
    EnvAliasTables tables;
    tables.width = width;
    tables.height = height;
    tables.rowQ.resize(width * height);
    tables.rowAlias.resize(width * height);
    tables.colQ.resize(height);
    tables.colAlias.resize(height);
    tables.pdfMarginal.resize(height);
    tables.pdfRow.resize(width * height);

    std::vector<float> rowSums(height, 0.0f);
    std::vector<float> rowWeights(width, 0.0f);

    // luminace有的地方为0，将军说了，所有的像素都有可能被采样到。
    // 先统计最大值
    //for (int i = 0; i < width * height; i++) {
    //    luminance[i] = std::pow(luminance[i],2);
    //}
    

    //for (int i = 0; i < width * height; i++) {
    //    luminance[i] += 0.001 * maxValue;
    //}

    for (int r = 0; r < height; ++r)
    {
        float sum = 0.0f;

        for (int c = 0; c < width; ++c)
        {
            float w = luminance[r * width + c];
            if (!std::isfinite(w) || w < 0.0f)
                w = 0.0f;
            rowWeights[c] = w;
            sum += w;
        }

        rowSums[r] = sum;

        AliasTable rowTable = BuildAliasTable1D(rowWeights.data(), width);

        for (int c = 0; c < width; ++c)
        {
            tables.rowQ[r * width + c] = rowTable.q[c];
            tables.rowAlias[r * width + c] = rowTable.alias[c];
        }
        // 拷贝行的pdf
        for (int c = 0; c < width; c++) {
            if (rowSums[r] > 1e-7f) {
                tables.pdfRow[r * width + c] = rowWeights[c] / rowSums[r];
            }
            else {
                tables.pdfRow[r * width + c] = 1.0f / (float)width;
            }
        }
    }

    AliasTable colTable = BuildAliasTable1D(rowSums.data(), height);

    for (int r = 0; r < height; ++r)
    {
        tables.colQ[r] = colTable.q[r];
        tables.colAlias[r] = colTable.alias[r];
    }
    // 拷贝边缘pdf
    float MarginalSum = 0;
    for (int r = 0; r < height; r++) {
        MarginalSum += rowSums[r];
    }
    for (int r = 0; r < height; r++) {
        if (MarginalSum > 1e-7f) {
            tables.pdfMarginal[r] = rowSums[r] / MarginalSum;
        }
        else {
            tables.pdfMarginal[r] = 1.0f / (float)height;
        }
    }
    return tables;
}

inline int SampleAlias1D(
    const float* q,
    const int* alias,
    int n,
    float u0,
    float u1)
{
    int idx = std::min(int(u0 * n), n - 1);
    return (u1 < q[idx]) ? idx : alias[idx];
}

inline SampledPoint SampleEnvAlias(
    const EnvAliasTables& tables,
    float u0, float u1,
    float u2, float u3)
{
    int row = SampleAlias1D(
        tables.colQ.data(),
        tables.colAlias.data(),
        tables.height,
        u0, u1);

    const float* rowQ = &tables.rowQ[row * tables.width];
    const int* rowAlias = &tables.rowAlias[row * tables.width];

    int col = SampleAlias1D(
        rowQ,
        rowAlias,
        tables.width,
        u2, u3);

    return { row, col };
}

// ============================================================
// Device -> Host copy
// ============================================================

inline std::vector<float> CopyFloatImageFromDevice(const float* d_data, int width, int height)
{
    std::vector<float> h_data(width * height);
    CUDA_CHECK(cudaMemcpy(h_data.data(),
        d_data,
        sizeof(float) * width * height,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    return h_data;
}

// ============================================================
// 把 luminance 做成可视化图，并把采样点标红
// ============================================================

inline std::vector<float4> MakeMarkedImageFromLuminance(
    const float* luminance,
    int width,
    int height,
    const std::vector<SampledPoint>& points)
{
    float maxVal = 0.0f;
    for (int i = 0; i < width * height; ++i)
        maxVal = std::max(maxVal, luminance[i]);

    if (!(maxVal > 0.0f) || !std::isfinite(maxVal))
        maxVal = 1.0f;

    std::vector<float4> image(width * height);

    for (int r = 0; r < height; ++r)
    {
        for (int c = 0; c < width; ++c)
        {
            int idx = r * width + c;
            float g = luminance[idx] / maxVal;
            g = std::max(0.0f, std::min(g, 1.0f));
            image[idx] = make_float4(g, g, g, 1.0f);
        }
    }

    for (const auto& p : points)
    {
        for (int dr = -1; dr <= 1; ++dr)
        {
            for (int dc = -1; dc <= 1; ++dc)
            {
                int rr = p.row + dr;
                int cc = p.col + dc;

                if (rr < 0 || rr >= height || cc < 0 || cc >= width)
                    continue;

                int idx = rr * width + cc;
                image[idx] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
            }
        }
    }

    return image;
}