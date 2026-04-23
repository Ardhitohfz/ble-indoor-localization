#pragma once
#include <Arduino.h>
#include "app_config.h"
#include "payload.h"

inline float rssiToDistance(int rssi, uint8_t anchor_idx)
{
    if (anchor_idx >= 4)
        return -1.0f;

    constexpr float d_ref = 1.0f;
    float A_ref = A_REF[anchor_idx];
    float N = N_PLE[anchor_idx];

    float exponent = (A_ref - rssi) / (10.0f * N);
    float distance = d_ref * pow(10.0f, exponent);

    if (distance < 0.5f)
        distance = 0.5f;
    if (distance > 20.0f)
        distance = 20.0f;

    return distance;
}

PositionResult calculatePosition(const AnchorRSSI anchors[4])
{
    PositionResult result = {0, 0, 0, false, 0};

    int validCount = 0;
    int validIndices[4];
    for (int i = 0; i < 4; i++)
    {
        if (anchors[i].valid)
        {
            validIndices[validCount] = i;
            validCount++;
        }
    }
    result.num_anchors = validCount;

    if (validCount < 3)
    {
        return result;
    }

    if (validCount == 3)
    {
        int idx0 = validIndices[0];
        int idx1 = validIndices[1];
        int idx2 = validIndices[2];

        float dx1 = ANCHOR_X[idx1] - ANCHOR_X[idx0];
        float dy1 = ANCHOR_Y[idx1] - ANCHOR_Y[idx0];
        float dx2 = ANCHOR_X[idx2] - ANCHOR_X[idx0];
        float dy2 = ANCHOR_Y[idx2] - ANCHOR_Y[idx0];

        float cross = fabs(dx1 * dy2 - dy1 * dx2);

        if (cross < 2.0f)
        {
            result.error_estimate = 10.0f;
        }
    }

    float distances[4];
    for (int i = 0; i < validCount; i++)
    {
        int idx = validIndices[i];
        distances[i] = anchors[idx].distance;
    }

    int ref_idx = validIndices[0];
    float x0 = ANCHOR_X[ref_idx], y0 = ANCHOR_Y[ref_idx], d0 = distances[0];

    float A11 = 0, A12 = 0, A22 = 0;
    float b1 = 0, b2 = 0;

    for (int i = 1; i < validCount; i++)
    {
        int idx = validIndices[i];
        float xi = ANCHOR_X[idx];
        float yi = ANCHOR_Y[idx];
        float di = distances[i];

        float dx = xi - x0;
        float dy = yi - y0;
        float K = xi * xi + yi * yi - x0 * x0 - y0 * y0 - di * di + d0 * d0;

        float weight = 1.0f / (di * di + 0.1f);
        if (weight > 10.0f)
            weight = 10.0f;

        A11 += weight * dx * dx;
        A12 += weight * dx * dy;
        A22 += weight * dy * dy;
        b1 += weight * dx * K;
        b2 += weight * dy * K;
    }

    float det = A11 * A22 - A12 * A12;

    if (fabs(det) < 1e-6f)
    {
        result.x = 0;
        result.y = 0;
        float total_weight = 0;

        for (int i = 0; i < validCount; i++)
        {
            int idx = validIndices[i];
            float weight = 1.0f / (distances[i] + 0.5f);
            if (weight > 5.0f)
                weight = 5.0f;

            result.x += ANCHOR_X[idx] * weight;
            result.y += ANCHOR_Y[idx] * weight;
            total_weight += weight;
        }

        if (total_weight > 0)
        {
            result.x /= total_weight;
            result.y /= total_weight;
        }

        result.error_estimate = 5.0f;
        result.valid = true;
        return result;
    }

    result.x = (b1 * A22 - b2 * A12) / (2.0f * det);
    result.y = (b2 * A11 - b1 * A12) / (2.0f * det);

    float residual_sum = 0;
    for (int i = 0; i < validCount; i++)
    {
        int idx = validIndices[i];
        float dx = result.x - ANCHOR_X[idx];
        float dy = result.y - ANCHOR_Y[idx];
        float calc_dist = sqrt(dx * dx + dy * dy);
        float error = calc_dist - distances[i];
        residual_sum += error * error;
    }
    result.error_estimate = sqrt(residual_sum / validCount);

    if (validCount == 3)
    {
        result.error_estimate *= 1.5f;
    }

    if (result.x < 0)
        result.x = 0;
    if (result.x > AREA_W)
        result.x = AREA_W;
    if (result.y < 0)
        result.y = 0;
    if (result.y > AREA_H)
        result.y = AREA_H;

    result.valid = true;
    return result;
}

inline String positionToGridCell(float x, float y)
{
    int col = static_cast<int>(x / CELL_W);
    int row = static_cast<int>(y / CELL_H);

    if (col < 0)
        col = 0;
    if (col >= GRID_NX)
        col = GRID_NX - 1;
    if (row < 0)
        row = 0;
    if (row >= GRID_NY)
        row = GRID_NY - 1;

    char cell[4];
    cell[0] = 'A' + col;
    cell[1] = '1' + row;
    cell[2] = '\0';

    return String(cell);
}

class RSSIFilter
{
private:
    static constexpr int WINDOW_SIZE = 5;
    int values[WINDOW_SIZE];
    int index;
    int count;

public:
    RSSIFilter() : index(0), count(0)
    {
        for (int i = 0; i < WINDOW_SIZE; i++)
            values[i] = 127;
    }

    void addSample(int rssi)
    {
        values[index] = rssi;
        index = (index + 1) % WINDOW_SIZE;
        if (count < WINDOW_SIZE)
            count++;
    }

    int getAverage() const
    {
        if (count == 0)
            return 127;
        int sum = 0;
        for (int i = 0; i < count; i++)
        {
            sum += values[i];
        }
        return sum / count;
    }

    void reset()
    {
        index = 0;
        count = 0;
        for (int i = 0; i < WINDOW_SIZE; i++)
            values[i] = 127;
    }
};