#include <Arduino.h>
#include <NimBLEDevice.h>
#include <algorithm>
#include <cmath>
#include "app_config.h"
#include "payload.h"
#include "telemetry.h"
#include "math_loc.h"
#include "ble_callbacks.h"

#ifndef ANCHOR_INDEX
#define ANCHOR_INDEX 0
#warning "ANCHOR_INDEX not defined, using 0"
#endif

#ifndef ANCHOR_NAME
#define ANCHOR_NAME "ANCHOR_UNKNOWN"
#warning "ANCHOR_NAME not defined"
#endif

#ifndef BEACON_NAME
#define BEACON_NAME "VEH_BEACON"
#warning "BEACON_NAME not defined"
#endif

#ifndef ROLE_ANCHOR
#define ROLE_ANCHOR
#endif

#if ANCHOR_INDEX < 0 || ANCHOR_INDEX > 3
#error "ANCHOR_INDEX must be 0-3"
#endif

static_assert(ANCHOR_SCAN_INTERVAL_MS >= 1000,
              "ANCHOR_SCAN_INTERVAL_MS must be >= 1000ms to prevent scan overlap");

static_assert(ANCHOR_SCAN_INTERVAL_MS <= 60000,
              "ANCHOR_SCAN_INTERVAL_MS should be <= 60s for reasonable update rate");

NimBLEServer *pServer = nullptr;
NimBLEService *pService = nullptr;
NimBLECharacteristic *pDataChar = nullptr;
NimBLECharacteristic *pStatsChar = nullptr;
NimBLECharacteristic *pControlChar = nullptr;
NimBLEAdvertising *pAdvertising = nullptr;
NimBLEScan *pScan = nullptr;
RSSIFilter rssiFilter;
static uint8_t advSequence = 0;

std::atomic<bool> sinkConnected(false);
std::atomic<bool> isScanning(false);
std::atomic<bool> scanRequested(false);
std::atomic<int8_t> lastAvgRSSI(-128);

static constexpr size_t kMaxSamplesPerScan = 64;
static int scanResults[kMaxSamplesPerScan];
static size_t scanResultsCount = 0;
static size_t scanResultsWriteIdx = 0;
SemaphoreHandle_t scanResultsMutex = nullptr;
static NimBLEAddress targetBeaconAddr;

struct ConnectionMetrics
{
    uint32_t sink_connection_count;
};

ConnectionMetrics connMetrics = {0};

class ScanCallbacks : public NimBLEScanCallbacks
{
    void onResult(const NimBLEAdvertisedDevice *advertisedDevice) override
    {
        int rssi = advertisedDevice->getRSSI();

#ifdef ENABLE_BLE_SCAN_DIAGNOSTICS
        String deviceName = advertisedDevice->haveName()
                                ? advertisedDevice->getName().c_str()
                                : "<no name>";
        LOG_I("ANCHOR", "BLE | MAC: %s | Name: %-20s | RSSI: %3d dBm",
              advertisedDevice->getAddress().toString().c_str(), deviceName.c_str(), rssi);
#endif

        bool isBeacon = (advertisedDevice->getAddress() == targetBeaconAddr);
        if (!isBeacon && advertisedDevice->haveName())
        {
            isBeacon = (advertisedDevice->getName() == BEACON_NAME);
        }

        if (!isBeacon)
            return;

        LOG_D("ANCHOR", "BEACON DETECTED | %s | RSSI: %d dBm",
              advertisedDevice->getAddress().toString().c_str(), rssi);

        if (!isValidRSSI(rssi))
        {
            LOG_W("ANCHOR", "Invalid RSSI value: %d dBm (out of range)", rssi);
            return;
        }

        if (xSemaphoreTake(scanResultsMutex, pdMS_TO_TICKS(100)) != pdTRUE)
        {
            LOG_W("ANCHOR", "Failed to acquire mutex for RSSI sample");
            return;
        }

        scanResults[scanResultsWriteIdx] = rssi;
        scanResultsWriteIdx = (scanResultsWriteIdx + 1) % kMaxSamplesPerScan;
        if (scanResultsCount < kMaxSamplesPerScan)
            scanResultsCount++;

        xSemaphoreGive(scanResultsMutex);
    }
};

struct StatisticsResult
{
    int8_t mean;
    int8_t median;
    int8_t std_dev;
    int8_t min;
    int8_t max;
    int8_t q25;
    int8_t q75;
    uint8_t outliers;
    bool valid;
};

StatisticsResult calculateStatistics(const int *samples, size_t n)
{
    StatisticsResult result = {0, 0, 0, 0, 0, 0, 0, 0, false};

    if (n < 3)
    {
        LOG_W("ANCHOR", "Insufficient samples for statistics: %d (need >= 3)", n);
        return result;
    }

    static int sortedData[kMaxSamplesPerScan];

    for (size_t i = 0; i < n; i++)
        sortedData[i] = samples[i];

    int min_val = sortedData[0];
    int max_val = sortedData[0];
    long sum = 0;

    for (size_t i = 0; i < n; i++)
    {
        int val = sortedData[i];
        if (val < min_val)
            min_val = val;
        if (val > max_val)
            max_val = val;
        sum += val;
    }

    result.min = static_cast<int8_t>(min_val);
    result.max = static_cast<int8_t>(max_val);

    long mean_calc = sum / static_cast<long>(n);
    LOG_D("ANCHOR", "Mean calc: sum=%ld, n=%zu, mean_raw=%ld (samples[0]=%d)",
          sum, n, mean_calc, samples[0]);

    if (mean_calc > 0)
    {
        LOG_W("ANCHOR", "BUG: Mean is POSITIVE (%ld)! Setting to 0", mean_calc);
        mean_calc = 0;
    }
    if (mean_calc < -127)
    {
        LOG_W("ANCHOR", "Mean underflow (%ld), clamping to -127", mean_calc);
        mean_calc = -127;
    }
    result.mean = static_cast<int8_t>(mean_calc);

    std::sort(sortedData, sortedData + n);

    size_t q25_idx = n / 4;
    size_t median_idx = n / 2;
    size_t q75_idx = (3 * n) / 4;

    result.q25 = static_cast<int8_t>(sortedData[q25_idx]);
    result.median = static_cast<int8_t>(sortedData[median_idx]);
    result.q75 = static_cast<int8_t>(sortedData[q75_idx]);

    float iqr = result.q75 - result.q25;
    float lower_bound = result.q25 - 1.5f * iqr;
    float upper_bound = result.q75 + 1.5f * iqr;

    float variance = 0.0f;
    uint8_t outlier_count = 0;

    for (size_t i = 0; i < n; i++)
    {
        int val = samples[i];

        float diff = val - result.mean;
        variance += diff * diff;

        if (val < lower_bound || val > upper_bound)
            outlier_count++;
    }

    variance /= n;
    float std_dev_float = std::sqrt(variance);

    float std_dev_scaled = std_dev_float * 10.0f;
    if (std_dev_scaled > 127.0f)
        std_dev_scaled = 127.0f;
    if (std_dev_scaled < 0.0f)
        std_dev_scaled = 0.0f;
    result.std_dev = static_cast<int8_t>(std_dev_scaled);

    result.outliers = outlier_count;

    result.valid = true;
    LOG_D("ANCHOR", "Stats | mean=%d median=%d std=%.1f (clamped=%d) min=%d max=%d q25=%d q75=%d outliers=%d",
          result.mean, result.median, std_dev_float, result.std_dev,
          result.min, result.max, result.q25, result.q75, result.outliers);

    // Final validation check
    if (result.mean > 0)
    {
        LOG_E("ANCHOR", "CRITICAL BUG: Mean is still positive after clamping! mean=%d", result.mean);
    }
    if (result.std_dev < 0)
    {
        LOG_E("ANCHOR", "CRITICAL BUG: Std dev is negative after clamping! std=%d", result.std_dev);
    }

    return result;
}

bool performBlockingBeaconScan(uint8_t maxRetries);
bool performBootScan();
void updateAdvertisingPayload(int8_t rssiValue, uint32_t timestampMs);

void updateAdvertisingPayload(int8_t rssiValue, uint32_t timestampMs)
{
    if (pAdvertising == nullptr)
    {
        LOG_E("ANCHOR", "Advertising handle not initialized");
        return;
    }

    NimBLEAdvertisementData advData;
    advData.setName(ANCHOR_NAME);

    uint8_t mfg[9];
    mfg[0] = CUSTOM_COMPANY_ID & 0xFF;
    mfg[1] = (CUSTOM_COMPANY_ID >> 8) & 0xFF;
    mfg[2] = ANCHOR_INDEX;
    mfg[3] = (uint8_t)rssiValue;
    mfg[4] = timestampMs & 0xFF;
    mfg[5] = (timestampMs >> 8) & 0xFF;
    mfg[6] = (timestampMs >> 16) & 0xFF;
    mfg[7] = (timestampMs >> 24) & 0xFF;
    mfg[8] = advSequence++;

    advData.setManufacturerData(
        std::string(reinterpret_cast<char *>(mfg), sizeof(mfg)));

    pAdvertising->setAdvertisementData(advData);
}

class ControlCharCallbacks : public NimBLECharacteristicCallbacks
{
    void onWrite(NimBLECharacteristic *pCharacteristic, NimBLEConnInfo &connInfo) override
    {
        std::string value = pCharacteristic->getValue();

        if (value.length() == 0)
        {
            LOG_W("ANCHOR", "Control char: Empty write received");
            return;
        }

        uint8_t cmd = (uint8_t)value[0];

        switch (cmd)
        {
        case CMD_TRIGGER_SCAN:
        {
            LOG_I("ANCHOR", "Control: TRIGGER_SCAN command received");

            if (scanRequested)
            {
                LOG_W("ANCHOR", "Scan already requested, ignoring duplicate");
                uint8_t resp = RESP_SCAN_IN_PROGRESS;
                pCharacteristic->setValue(&resp, 1);
                pCharacteristic->notify();
                return;
            }

            if (isScanning)
            {
                LOG_W("ANCHOR", "Scan in progress, cannot start new scan");
                uint8_t resp = RESP_SCAN_IN_PROGRESS;
                pCharacteristic->setValue(&resp, 1);
                pCharacteristic->notify();
                return;
            }

            scanRequested = true;

            uint8_t resp = RESP_SCAN_STARTED;
            pCharacteristic->setValue(&resp, 1);
            pCharacteristic->notify();

            LOG_I("ANCHOR", "[OK] Scan request accepted, will execute in main loop");
            break;
        }

        case CMD_STATUS_QUERY:
        {
            LOG_D("ANCHOR", "Control: STATUS_QUERY command received");

            uint8_t status;
            if (isScanning)
            {
                status = RESP_SCAN_IN_PROGRESS;
            }
            else if (lastAvgRSSI != -128)
            {
                status = RESP_SCAN_COMPLETE;
            }
            else
            {
                status = RESP_SCAN_FAILED;
            }

            pCharacteristic->setValue(&status, 1);
            pCharacteristic->notify();
            break;
        }

        default:
            LOG_W("ANCHOR", "Control char: Unknown command: 0x%02X", cmd);
            break;
        }
    }
};

void setup()
{
    Serial.begin(115200);
    delay(SERIAL_INIT_DELAY_MS);

    targetBeaconAddr = NimBLEAddress(BEACON_MAC, BLE_ADDR_PUBLIC);

    scanResultsMutex = xSemaphoreCreateMutex();
    if (scanResultsMutex == nullptr)
    {
        LOG_E("ANCHOR", "[FATAL] Failed to create mutex!");
        LOG_E("ANCHOR", "Rebooting in %lu seconds to attempt recovery...", REBOOT_DELAY_MS / 1000);
        delay(REBOOT_DELAY_MS);
        ESP.restart();
    }
    LOG_I("ANCHOR", "[OK] Thread safety mutex created");

    printSystemInfo("ANCHOR");
    LOG_I("ANCHOR", "========================================");
    LOG_I("ANCHOR", "Name: %s (Index: %d)", ANCHOR_NAME, ANCHOR_INDEX);
    LOG_I("ANCHOR", "Target Beacon: %s", BEACON_NAME);
    LOG_I("ANCHOR", "========================================");

    if (ANCHOR_INDEX < 0 || ANCHOR_INDEX > 3)
    {
        LOG_E("ANCHOR", "[FATAL] CONFIGURATION ERROR: Invalid ANCHOR_INDEX: %d", ANCHOR_INDEX);
        LOG_E("ANCHOR", "Valid range: 0-3. Check app_config.h!");
        LOG_E("ANCHOR", "System halted. Fix configuration and re-flash firmware.");
        while (1)
            delay(ERROR_HALT_LOOP_DELAY_MS);
    }

    LOG_I("ANCHOR", "Initializing NimBLE device...");
    NimBLEDevice::init(ANCHOR_NAME);

    delay(BLE_STACK_SETTLE_DELAY_MS);

    LOG_I("ANCHOR", "BLE initialized (after 1s settle time)");
    LOG_I("ANCHOR", "MAC: %s", NimBLEDevice::getAddress().toString().c_str());

#ifdef BLE_SECURITY_ENABLED
    LOG_I("ANCHOR", "Configuring BLE security...");

    NimBLEDevice::setSecurityAuth(
        true,
        false,
        true);

    NimBLEDevice::setSecurityIOCap(BLE_HS_IO_NO_INPUT_OUTPUT);

    LOG_I("ANCHOR", "[OK] Security configured: Bonding + Secure Connections");
#else
    LOG_W("ANCHOR", "[WARN] Security DISABLED (set BLE_SECURITY_ENABLED in app_config.h)");
#endif

    LOG_I("ANCHOR", "Creating BLE server (Peripheral role)...");
    pServer = NimBLEDevice::createServer();

    LOG_I("ANCHOR", "Creating service: %s", SVC_UUID);
    pService = pServer->createService(SVC_UUID);

    LOG_I("ANCHOR", "Creating characteristic: %s", CH_DATA_UUID);
    pDataChar = pService->createCharacteristic(
        CH_DATA_UUID,
        NIMBLE_PROPERTY::READ);

    LOG_I("ANCHOR", "[OK] Data characteristic created (READ only, cached RSSI)");

    uint8_t init_data[ANCHOR_PAYLOAD_SIZE];
    encodeAnchorPayload(init_data, ANCHOR_INDEX, -128, 0);
    pDataChar->setValue(init_data, ANCHOR_PAYLOAD_SIZE);

    LOG_I("ANCHOR", "Creating control characteristic: %s", CH_CONTROL_UUID);
    pControlChar = pService->createCharacteristic(
        CH_CONTROL_UUID,
        NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::NOTIFY);

    pControlChar->setCallbacks(new ControlCharCallbacks());

    LOG_I("ANCHOR", "[OK] Control characteristic created (on-demand scan capability)");
    LOG_I("ANCHOR", "   Sink can write CMD_TRIGGER_SCAN (0x01) for fresh RSSI");

    pStatsChar = pService->createCharacteristic(
        CH_STATS_UUID,
        NIMBLE_PROPERTY::READ);

    uint8_t statsBuffer[ANCHOR_STATS_PAYLOAD_SIZE] = {0};
    encodeAnchorStatsPayload(statsBuffer, ANCHOR_INDEX, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    pStatsChar->setValue(statsBuffer, ANCHOR_STATS_PAYLOAD_SIZE);

    LOG_I("ANCHOR", "[OK] Statistics characteristic created (8 features for ML)");
    LOG_I("ANCHOR", "   Features: mean, median, std, min, max, q25, q75, outliers");

    pService->start();
    LOG_I("ANCHOR", "Service started");

    LOG_I("ANCHOR", "Configuring advertising...");
    pAdvertising = NimBLEDevice::getAdvertising();

    pAdvertising->setConnectableMode(BLE_GAP_CONN_MODE_UND);

    pAdvertising->enableScanResponse(false);

    updateAdvertisingPayload(-128, 0);

    pAdvertising->setMinInterval(BLE_ADV_INTERVAL_UNITS);
    pAdvertising->setMaxInterval(BLE_ADV_INTERVAL_UNITS);

    LOG_I("ANCHOR", "Advertising interval: %u ms", BLE_ADV_INTERVAL_MS);
    LOG_I("ANCHOR", "Connection params will be set in onConnect() callback");

    unsigned long timeSlotOffset = TIME_SLOT_OFFSET_MS[ANCHOR_INDEX];
    if (timeSlotOffset > 0)
    {
        LOG_I("ANCHOR", "[TIMING] Applying time-slot offset: %lu ms", timeSlotOffset);
        LOG_I("ANCHOR", "Time slot: [%lu ms - %lu ms]",
              timeSlotOffset, timeSlotOffset + TIME_SLOT_DURATION_MS);
        delay(timeSlotOffset);
    }
    else
    {
        LOG_I("ANCHOR", "Time slot: [0 ms - %lu ms] (First slot, no delay)", TIME_SLOT_DURATION_MS);
    }

    LOG_I("ANCHOR", "[OK] Time-slot staggering: ENABLED (v6.1 collision prevention)");

    LOG_I("ANCHOR", "Starting advertising...");
    if (pAdvertising->start())
    {
        LOG_I("ANCHOR", "[OK] Advertising started successfully");
    }
    else
    {
        LOG_E("ANCHOR", "[FATAL] Advertising failed to start!");
        LOG_E("ANCHOR", "Rebooting in %lu seconds to attempt recovery...", REBOOT_DELAY_MS / 1000);
        delay(REBOOT_DELAY_MS);
        ESP.restart();
    }

    LOG_I("ANCHOR", "Initializing BLE scanner (Central role)...");
    pScan = NimBLEDevice::getScan();

    pScan->setScanCallbacks(new ScanCallbacks(), true);

    pScan->setActiveScan(true);
    pScan->setInterval(BLE_SCAN_INTERVAL_MS);
    pScan->setWindow(BLE_SCAN_WINDOW_MS);
    pScan->setDuplicateFilter(false);

    LOG_I("ANCHOR", "[OK] Scanner initialized");

    pServer->setCallbacks(new AnchorServerCallbacks(
        &sinkConnected,
        &connMetrics.sink_connection_count,
        &isScanning,
        pScan,
        pAdvertising));

    validateConfiguration("ANCHOR");

#ifdef PARALLEL_MODE

    LOG_I("ANCHOR", "========================================");
    LOG_I("ANCHOR", "PARALLEL MODE: Quick boot scan for initial RSSI");
    LOG_I("ANCHOR", "Sink can still trigger on-demand scan via control char");
    LOG_I("ANCHOR", "========================================");

    bool beaconDetected = performBlockingBeaconScan(1);
    if (!beaconDetected)
    {
        LOG_W("ANCHOR", "Initial scan failed; payload remains INVALID (-128)");
    }

#else

    LOG_I("ANCHOR", "========================================");
    LOG_I("ANCHOR", "NORMAL MODE: Starting one-time beacon scan at boot...");
    LOG_I("ANCHOR", "========================================");

    int retryCount = 0;
    bool beaconDetected = false;
    const int MAX_RETRIES = BOOT_SCAN_MAX_RETRIES;

    while (!beaconDetected && retryCount < MAX_RETRIES)
    {
        LOG_I("ANCHOR", "Scan attempt %d/%d", retryCount + 1, MAX_RETRIES);

        beaconDetected = performBootScan();

        if (!beaconDetected && retryCount < MAX_RETRIES - 1)
        {
            LOG_W("ANCHOR", "Beacon not found, retrying in 1 second...");
            delay(BOOT_SCAN_RETRY_DELAY_MS);
        }

        retryCount++;
    }

    LOG_I("ANCHOR", "========================================");
    if (beaconDetected)
    {
        LOG_I("ANCHOR", "Boot scan SUCCESSFUL; RSSI cached: %d dBm", lastAvgRSSI);
        LOG_I("ANCHOR", "   Anchor will serve this value to Sink (static environment)");
    }
    else
    {
        LOG_E("ANCHOR", "Boot scan FAILED after %d attempts", MAX_RETRIES);
        LOG_E("ANCHOR", "Anchor will serve INVALID marker (-128) to Sink");
        LOG_E("ANCHOR", "Possible causes:");
        LOG_E("ANCHOR", "1) Beacon not powered on");
        LOG_E("ANCHOR", "2) Beacon out of range");
        LOG_E("ANCHOR", "3) MAC address mismatch in app_config.h");
        LOG_E("ANCHOR", "Fix issue and reboot anchor to retry scan");
    }
    LOG_I("ANCHOR", "========================================");
#endif

    LOG_I("ANCHOR", "========================================");
    LOG_I("ANCHOR", "ANCHOR READY - ACCEPTING CONNECTIONS");
    LOG_I("ANCHOR", "Peripheral: Serving Sink (cached RSSI)");
    LOG_I("ANCHOR", "Central: Boot scan complete");
    LOG_I("ANCHOR", "Status: IDLE (ready for on-demand scan)");
    LOG_I("ANCHOR", "========================================");
    LOG_I("ANCHOR", "Anchor is ready for Sink connections.");
    LOG_I("ANCHOR", "Safe to run: python3 test_anchors.py");
}

bool performBlockingBeaconScan(uint8_t maxRetries)
{
    for (uint8_t attempt = 1; attempt <= maxRetries; attempt++)
    {
        LOG_I("ANCHOR", "Beacon scan attempt %d/%d (duration: %ds)...",
              attempt, maxRetries, ONDEMAND_SCAN_DURATION_SEC);

        rssiFilter.reset();
        if (xSemaphoreTake(scanResultsMutex, pdMS_TO_TICKS(100)) == pdTRUE)
        {
            // Reset circular buffer
            scanResultsCount = 0;
            scanResultsWriteIdx = 0;
            xSemaphoreGive(scanResultsMutex);
        }
        else
        {
            LOG_E("ANCHOR", "[ERROR] Failed to acquire mutex for clearing scan results");
            if (attempt < maxRetries)
            {
                delay(ONDEMAND_SCAN_RETRY_DELAY_MS);
                continue;
            }
            return false;
        }

        isScanning = true;

        const uint32_t scanDurationMs = ONDEMAND_SCAN_DURATION_SEC * 1000;
        NimBLEScanResults results = pScan->getResults(scanDurationMs, false);

        isScanning = false;

        int sampleCount = 0;
        if (xSemaphoreTake(scanResultsMutex, pdMS_TO_TICKS(100)) == pdTRUE)
        {
            sampleCount = scanResultsCount;

            if (sampleCount > 0)
            {
                int tempSamples[kMaxSamplesPerScan];

                if (scanResultsCount == kMaxSamplesPerScan && scanResultsWriteIdx > 0)
                {
                    size_t firstPartSize = kMaxSamplesPerScan - scanResultsWriteIdx;
                    for (size_t i = 0; i < firstPartSize; i++)
                        tempSamples[i] = scanResults[scanResultsWriteIdx + i];
                    for (size_t i = 0; i < scanResultsWriteIdx; i++)
                        tempSamples[firstPartSize + i] = scanResults[i];
                }
                else
                {
                    for (size_t i = 0; i < sampleCount; i++)
                        tempSamples[i] = scanResults[i];
                }

                for (int i = 0; i < sampleCount; i++)
                {
                    rssiFilter.addSample(tempSamples[i]);
                }
                lastAvgRSSI = (int8_t)rssiFilter.getAverage();

                StatisticsResult stats = calculateStatistics(tempSamples, sampleCount);

                xSemaphoreGive(scanResultsMutex);

                uint8_t data[ANCHOR_PAYLOAD_SIZE];
                encodeAnchorPayload(data, ANCHOR_INDEX, lastAvgRSSI, millis());
                pDataChar->setValue(data, ANCHOR_PAYLOAD_SIZE);
                updateAdvertisingPayload(lastAvgRSSI, millis());

                if (stats.valid && pStatsChar != nullptr)
                {
                    uint8_t statsData[ANCHOR_STATS_PAYLOAD_SIZE];
                    encodeAnchorStatsPayload(
                        statsData, ANCHOR_INDEX,
                        stats.mean, stats.median, stats.std_dev,
                        stats.min, stats.max, stats.q25, stats.q75,
                        stats.outliers, millis());
                    pStatsChar->setValue(statsData, ANCHOR_STATS_PAYLOAD_SIZE);
                    LOG_D("ANCHOR", "[OK] Statistics characteristic updated");
                }

                LOG_I("ANCHOR", "[OK] Beacon detected! Samples: %d, Avg RSSI: %d dBm",
                      sampleCount, lastAvgRSSI.load());

                return true;
            }

            xSemaphoreGive(scanResultsMutex);
        }
        else
        {
            LOG_E("ANCHOR", "[ERROR] Failed to acquire mutex for processing scan results");
        }

        LOG_W("ANCHOR", "[WARN] No beacon detected in attempt %d/%d", attempt, maxRetries);

        if (attempt < maxRetries)
        {
            LOG_I("ANCHOR", "Retrying in %dms...", ONDEMAND_SCAN_RETRY_DELAY_MS);
            delay(ONDEMAND_SCAN_RETRY_DELAY_MS);
        }
    }

    LOG_E("ANCHOR", "[ERROR] Beacon scan failed after %d attempts", maxRetries);
    lastAvgRSSI = -128;

    uint8_t data[ANCHOR_PAYLOAD_SIZE];
    encodeAnchorPayload(data, ANCHOR_INDEX, -128, millis());
    pDataChar->setValue(data, ANCHOR_PAYLOAD_SIZE);
    updateAdvertisingPayload(-128, millis());

    return false;
}

bool performBootScan()
{
    const uint8_t MAX_RETRIES = BOOT_SCAN_MAX_RETRIES;

    LOG_I("ANCHOR", "========================================");

    bool beaconDetected = performBlockingBeaconScan(MAX_RETRIES);

    LOG_I("ANCHOR", "========================================");
    if (beaconDetected)
    {
        LOG_I("ANCHOR", "Boot scan SUCCESSFUL; RSSI cached: %d dBm", lastAvgRSSI.load());
        LOG_I("ANCHOR", "Anchor will serve this value to Sink (static environment)");
        LOG_I("ANCHOR", "For fresh RSSI, Sink can write CMD_TRIGGER_SCAN to control char");
    }
    else
    {
        LOG_E("ANCHOR", "Boot scan FAILED after %d attempts", MAX_RETRIES);
        LOG_E("ANCHOR", "Anchor will serve INVALID marker (-128) to Sink");
        LOG_E("ANCHOR", "Possible causes:");
        LOG_E("ANCHOR", "1) Beacon not powered on");
        LOG_E("ANCHOR", "2) Beacon out of range");
        LOG_E("ANCHOR", "3) MAC address mismatch in app_config.h");
        LOG_E("ANCHOR", "Fix issue and reboot anchor to retry scan");
    }
    LOG_I("ANCHOR", "========================================");

    return beaconDetected;
}

void loop()
{

    if (scanRequested)
    {
        scanRequested = false;

        LOG_I("ANCHOR", "On-demand scan request detected");

        if (isScanning)
        {
            LOG_W("ANCHOR", "Cannot scan: Scan already in progress");

            uint8_t resp = RESP_SCAN_IN_PROGRESS;
            pControlChar->setValue(&resp, 1);
            pControlChar->notify();

            delay(ERROR_RECOVERY_DELAY_MS);
            return;
        }

        LOG_I("ANCHOR", "Executing on-demand beacon scan");
        bool success = performBlockingBeaconScan(ONDEMAND_SCAN_MAX_RETRIES);

        uint8_t resp = success ? RESP_SCAN_COMPLETE : RESP_SCAN_FAILED;
        pControlChar->setValue(&resp, 1);
        pControlChar->notify();

        if (success)
        {
            LOG_I("ANCHOR", "On-demand scan COMPLETE: %d dBm (fresh RSSI cached)",
                  lastAvgRSSI.load());
        }
        else
        {
            LOG_E("ANCHOR", "On-demand scan FAILED after %d attempts",
                  ONDEMAND_SCAN_MAX_RETRIES);
        }

        LOG_I("ANCHOR", "Back to IDLE mode");
    }

    delay(MAIN_LOOP_IDLE_DELAY_MS);
}
