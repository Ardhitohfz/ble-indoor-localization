#pragma once
#include <Arduino.h>

enum LogLevel
{
    LOG_ERROR = 0,
    LOG_WARN = 1,
    LOG_INFO = 2,
    LOG_DEBUG = 3
};

#ifndef GLOBAL_LOG_LEVEL
#define GLOBAL_LOG_LEVEL LOG_DEBUG
#endif

#define LOG_E(tag, fmt, ...)                                        \
    if (LOG_ERROR <= GLOBAL_LOG_LEVEL)                              \
    {                                                               \
        Serial.printf("[ERROR][%s] " fmt "\n", tag, ##__VA_ARGS__); \
    }

#define LOG_W(tag, fmt, ...)                                       \
    if (LOG_WARN <= GLOBAL_LOG_LEVEL)                              \
    {                                                              \
        Serial.printf("[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__); \
    }

#define LOG_I(tag, fmt, ...)                                       \
    if (LOG_INFO <= GLOBAL_LOG_LEVEL)                              \
    {                                                              \
        Serial.printf("[INFO][%s] " fmt "\n", tag, ##__VA_ARGS__); \
    }

#define LOG_D(tag, fmt, ...)                                        \
    if (LOG_DEBUG <= GLOBAL_LOG_LEVEL)                              \
    {                                                               \
        Serial.printf("[DEBUG][%s] " fmt "\n", tag, ##__VA_ARGS__); \
    }

inline void printSystemInfo(const char *role)
{
    Serial.println("==========================================");
    Serial.printf("ROLE: %s\n", role);
    Serial.printf("Chip: %s Rev %d\n", ESP.getChipModel(), ESP.getChipRevision());
    Serial.printf("Flash: %d MB\n", ESP.getFlashChipSize() / (1024 * 1024));
    Serial.printf("PSRAM: %d KB\n", ESP.getPsramSize() / 1024);
    Serial.printf("Free Heap: %d KB\n", ESP.getFreeHeap() / 1024);
    Serial.println("==========================================");
}

class PerformanceMonitor
{
private:
    unsigned long start_time;
    const char *operation_name;

public:
    PerformanceMonitor(const char *name) : operation_name(name)
    {
        start_time = millis();
    }

    ~PerformanceMonitor()
    {
        unsigned long duration = millis() - start_time;
        if (duration > 100) // >100ms = log warning
        {
            LOG_W("PERF", "%s took %lu ms", operation_name, duration);
        }
    }
};

// Monitors heap + fragmentation (>30% = risk) | Frag = 100% * (1 - max_block/free_heap)
inline void checkMemory(const char *tag, size_t threshold_bytes = 50000)
{
    size_t free_heap = ESP.getFreeHeap();
    size_t largest_block = ESP.getMaxAllocHeap();
    float fragmentation = 0.0f;
    if (free_heap > 0)
    {
        fragmentation = 100.0f * (1.0f - (float)largest_block / (float)free_heap);
    }

    if (free_heap < threshold_bytes)
    {
        LOG_W(tag, "Low memory: %zu bytes free (threshold: %zu)",
              free_heap, threshold_bytes);
        LOG_W(tag, "  Largest block: %zu bytes, fragmentation: %.1f%%",
              largest_block, fragmentation);
    }
    else if (fragmentation > 30.0f)
    {
        LOG_W(tag, "High memory fragmentation: %.1f%%",
              fragmentation);
        LOG_W(tag, "  Free: %zu bytes, largest block: %zu bytes",
              free_heap, largest_block);
    }
}

inline unsigned long timeElapsed(unsigned long current_ms, unsigned long previous_ms)
{
    return current_ms - previous_ms;
}

inline bool intervalPassed(unsigned long now, unsigned long lastTime, unsigned long interval)
{
    return timeElapsed(now, lastTime) >= interval;
}

inline bool isTimestampNewer(unsigned long ts1, unsigned long ts2)
{
    return (long)(ts1 - ts2) > 0;
}

inline bool isTimeDeltaValid(unsigned long delta_ms, unsigned long max_expected_ms)
{
    return (delta_ms < 0x80000000UL) && (delta_ms < max_expected_ms);
}