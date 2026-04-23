#include <Arduino.h>
#include <NimBLEDevice.h>
#include <atomic>
#include "app_config.h"
#include "payload.h"
#include "telemetry.h"
#include "ble_callbacks.h"

#ifndef DEVICE_NAME
#define DEVICE_NAME "VEH_BEACON"
#endif

#ifndef ROLE_BEACON
#define ROLE_BEACON
#endif

NimBLEAdvertising *pAdvertising = nullptr;

void updateBeaconTimestamp(uint32_t timestamp)
{
    if (pAdvertising == nullptr)
    {
        LOG_E("BEACON", "Advertising handle not initialized");
        return;
    }
    NimBLEAdvertisementData advData;
    advData.setPartialServices(NimBLEUUID(SVC_UUID));

    uint8_t syncData[BEACON_SYNC_PAYLOAD_SIZE];
    encodeBeaconSyncPayload(syncData, timestamp);
    advData.setManufacturerData(
        std::string(reinterpret_cast<char *>(syncData), BEACON_SYNC_PAYLOAD_SIZE));
    pAdvertising->setAdvertisementData(advData);

    LOG_D("BEACON", "Timestamp updated: %lu ms", (unsigned long)timestamp);
}

void setup()
{
    Serial.begin(115200);
    delay(SERIAL_INIT_DELAY_MS);

    printSystemInfo("BEACON");
    LOG_I("BEACON", "========================================");
    LOG_I("BEACON", "Device: %s", DEVICE_NAME);
    LOG_I("BEACON", "========================================");

    LOG_I("BEACON", "Initializing NimBLE device...");
    NimBLEDevice::init(DEVICE_NAME);

    LOG_I("BEACON", "BLE initialized");
    LOG_I("BEACON", "MAC: %s", NimBLEDevice::getAddress().toString().c_str());

    LOG_I("BEACON", "Configuring NON-CONNECTABLE advertising...");
    pAdvertising = NimBLEDevice::getAdvertising();

    pAdvertising->setConnectableMode(BLE_GAP_CONN_MODE_NON);
    LOG_I("BEACON", "Connectable mode: NON-CONNECTABLE (broadcast-only)");

    NimBLEAdvertisementData advData;
    advData.setPartialServices(NimBLEUUID(SVC_UUID));

    uint8_t syncData[BEACON_SYNC_PAYLOAD_SIZE];
    encodeBeaconSyncPayload(syncData, millis());
    advData.setManufacturerData(
        std::string(reinterpret_cast<char *>(syncData), BEACON_SYNC_PAYLOAD_SIZE));

    pAdvertising->setAdvertisementData(advData);
    LOG_I("BEACON", "Advertisement: service UUID + manufacturer timestamp (no name in ADV)");

    NimBLEAdvertisementData scanResponseData;
    scanResponseData.setName(DEVICE_NAME);
    pAdvertising->setScanResponseData(scanResponseData);
    LOG_I("BEACON", "[OK] Name also in SCAN RESPONSE (backward compatible)");

    pAdvertising->enableScanResponse(true);
    pAdvertising->setMinInterval(BLE_ADV_INTERVAL_UNITS);
    pAdvertising->setMaxInterval(BLE_ADV_INTERVAL_UNITS);

    LOG_I("BEACON", "Advertising interval: %u ms (fast for anchor sync)", BLE_ADV_INTERVAL_MS);

    LOG_I("BEACON", "Starting NON-CONNECTABLE advertising...");
    if (pAdvertising->start())
    {
        LOG_I("BEACON", "Advertising started successfully (NON-CONNECTABLE)");
        LOG_I("BEACON", "Service UUID: %s", SVC_UUID);
        LOG_I("BEACON", "Timestamp in manufacturer data (updated every %ums)", BEACON_ADV_UPDATE_MS);
        LOG_I("BEACON", "Beacon cannot be connected (true broadcast-only mode)");
    }
    else
    {
        LOG_E("BEACON", "Rebooting in %lu seconds to attempt recovery...", REBOOT_DELAY_MS / 1000);
        delay(REBOOT_DELAY_MS);
        ESP.restart();
    }

    validateConfiguration("BEACON");

    LOG_I("BEACON", "========================================");
    LOG_I("BEACON", "Beacon ready (NON-CONNECTABLE mode)");
    LOG_I("BEACON", "Always advertising - never stops");
    LOG_I("BEACON", "Timestamp update interval: %u ms", BEACON_ADV_UPDATE_MS);
    LOG_I("BEACON", "========================================");
}

void loop()
{
    uint32_t now = millis();
    static uint32_t lastSyncUpdate = 0;

    if (intervalPassed(now, lastSyncUpdate, BEACON_ADV_UPDATE_MS))
    {
        lastSyncUpdate = now;
        updateBeaconTimestamp(now);
    }

    static uint32_t lastMemCheck = 0;
    if (intervalPassed(now, lastMemCheck, MEMORY_CHECK_INTERVAL_MS))
    {
        lastMemCheck = now;
        checkMemory("BEACON");
        LOG_D("BEACON", "Uptime: %lu s, Mode: NON-CONNECTABLE (always advertising)",
              now / 1000);
    }

    delay(MAIN_LOOP_DELAY_MS);
}
