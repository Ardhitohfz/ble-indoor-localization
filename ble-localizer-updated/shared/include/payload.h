#pragma once
#include <Arduino.h>

struct __attribute__((packed)) AnchorPayload
{
    uint16_t company_id;
    uint8_t anchor_index;
    int8_t beacon_rssi;
    uint32_t timestamp_ms;
    uint8_t sync_status;
    int8_t drift_ms;
    uint16_t sync_count;
    uint32_t last_sync_time_ms;
};

struct PositionResult
{
    float x;
    float y;
    float error_estimate;
    bool valid;
    uint8_t num_anchors;
};

struct AnchorRSSI
{
    bool valid;
    int8_t rssi;
    float distance;
    uint32_t timestamp_ms;
};

static constexpr uint16_t CUSTOM_COMPANY_ID = 0xFFFF;
static constexpr size_t ANCHOR_PAYLOAD_SIZE = sizeof(AnchorPayload);

inline void encodeAnchorPayload(
    uint8_t *buffer,
    uint8_t anchor_idx,
    int8_t rssi,
    uint32_t ts,
    uint8_t sync_status = 0,
    int8_t drift_ms = 0,
    uint16_t sync_count = 0,
    uint32_t last_sync_ms = 0)
{
    AnchorPayload *payload = reinterpret_cast<AnchorPayload *>(buffer);
    payload->company_id = CUSTOM_COMPANY_ID;
    payload->anchor_index = anchor_idx;
    payload->beacon_rssi = rssi;
    payload->timestamp_ms = ts;
    payload->sync_status = sync_status;
    payload->drift_ms = drift_ms;
    payload->sync_count = sync_count & 0xFFFF;
    payload->last_sync_time_ms = last_sync_ms;
}

inline bool decodeAnchorPayload(const uint8_t *buffer, size_t len, AnchorPayload &out)
{
    if (len < ANCHOR_PAYLOAD_SIZE)
        return false;
    memcpy(&out, buffer, ANCHOR_PAYLOAD_SIZE);
    return (out.company_id == CUSTOM_COMPANY_ID && out.anchor_index < 4);
}

struct __attribute__((packed)) BeaconSyncPayload
{
    uint16_t company_id;
    uint32_t timestamp_ms;
};

static constexpr size_t BEACON_SYNC_PAYLOAD_SIZE = sizeof(BeaconSyncPayload);

inline void encodeBeaconSyncPayload(uint8_t *buffer, uint32_t ts)
{
    BeaconSyncPayload *payload = reinterpret_cast<BeaconSyncPayload *>(buffer);
    payload->company_id = CUSTOM_COMPANY_ID;
    payload->timestamp_ms = ts;
}

inline bool decodeBeaconSyncPayload(const uint8_t *buffer, size_t len, BeaconSyncPayload &out)
{
    if (len < BEACON_SYNC_PAYLOAD_SIZE)
        return false;
    memcpy(&out, buffer, BEACON_SYNC_PAYLOAD_SIZE);
    return (out.company_id == CUSTOM_COMPANY_ID);
}

struct __attribute__((packed)) AnchorStatsPayload
{
    uint16_t company_id;
    uint8_t anchor_index;
    int8_t mean;
    int8_t median;
    int8_t std_dev;
    int8_t min;
    int8_t max;
    int8_t q25;
    int8_t q75;
    uint8_t outliers;
    uint32_t timestamp_ms;
};

static constexpr size_t ANCHOR_STATS_PAYLOAD_SIZE = sizeof(AnchorStatsPayload);

inline void encodeAnchorStatsPayload(
    uint8_t *buffer,
    uint8_t anchor_idx,
    int8_t mean,
    int8_t median,
    int8_t std_dev,
    int8_t min,
    int8_t max,
    int8_t q25,
    int8_t q75,
    uint8_t outliers,
    uint32_t ts)
{
    AnchorStatsPayload *payload = reinterpret_cast<AnchorStatsPayload *>(buffer);
    payload->company_id = CUSTOM_COMPANY_ID;
    payload->anchor_index = anchor_idx;
    payload->mean = mean;
    payload->median = median;
    payload->std_dev = std_dev;
    payload->min = min;
    payload->max = max;
    payload->q25 = q25;
    payload->q75 = q75;
    payload->outliers = outliers;
    payload->timestamp_ms = ts;
}

inline bool decodeAnchorStatsPayload(const uint8_t *buffer, size_t len, AnchorStatsPayload &out)
{
    if (len < ANCHOR_STATS_PAYLOAD_SIZE)
        return false;
    memcpy(&out, buffer, ANCHOR_STATS_PAYLOAD_SIZE);
    return (out.company_id == CUSTOM_COMPANY_ID && out.anchor_index < 4);
}

inline bool isValidRSSI(int8_t rssi)
{
    return rssi >= -100 && rssi <= 0;
}