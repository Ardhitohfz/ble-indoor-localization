#pragma once
#include <Arduino.h>
#include "telemetry.h"

static constexpr char SVC_UUID[] = "f0e1d2c3-b4a5-9687-7856-453423120001";
static constexpr char CH_DATA_UUID[] = "f0e1d2c3-b4a5-9687-7856-453423120002";
static constexpr char CH_TIMESTAMP_UUID[] = "f0e1d2c3-b4a5-9687-7856-453423120003";
static constexpr char CH_CONTROL_UUID[] = "f0e1d2c3-b4a5-9687-7856-453423120004";
static constexpr char CH_STATS_UUID[] = "f0e1d2c3-b4a5-9687-7856-453423120005";

static constexpr uint8_t CMD_TRIGGER_SCAN = 0x01;
static constexpr uint8_t CMD_STATUS_QUERY = 0x02;
static constexpr uint8_t RESP_SCAN_STARTED = 0x11;
static constexpr uint8_t RESP_SCAN_COMPLETE = 0x12;
static constexpr uint8_t RESP_SCAN_FAILED = 0x13;
static constexpr uint8_t RESP_SCAN_IN_PROGRESS = 0x14;
static constexpr uint8_t RESP_SINK_BUSY = 0x15;

static constexpr uint32_t ONDEMAND_SCAN_DURATION_SEC = 3;
static constexpr uint8_t ONDEMAND_SCAN_MAX_RETRIES = 2;
static constexpr unsigned long ONDEMAND_SCAN_RETRY_DELAY_MS = 1000;

static_assert(sizeof(SVC_UUID) - 1 == 36, "Service UUID must be 36 characters (128-bit UUID format)");
static_assert(sizeof(CH_DATA_UUID) - 1 == 36, "Characteristic UUID must be 36 characters");
static_assert(sizeof(CH_TIMESTAMP_UUID) - 1 == 36, "Timestamp Characteristic UUID must be 36 characters");

static constexpr char BEACON_MAC[] = "fc:01:2c:d1:39:05";
static constexpr char ANCHOR_A_MAC[] = "d0:cf:13:15:78:2d";
static constexpr char ANCHOR_B_MAC[] = "d0:cf:13:15:9d:cd";
static constexpr char ANCHOR_C_MAC[] = "d0:cf:13:15:b0:fd";
static constexpr char ANCHOR_D_MAC[] = "d0:cf:13:14:f3:55";

static constexpr unsigned long ANCHOR_SCAN_INTERVAL_MS = 5000;
static constexpr unsigned long BEACON_RESCAN_DELAY_MS = 200;
static constexpr unsigned long SINK_SCAN_TIMEOUT_MS = 10000;

static constexpr uint16_t BLE_ADV_INTERVAL_MS = 100;
static constexpr uint16_t BLE_ADV_INTERVAL_UNITS = 160;
static constexpr uint32_t BEACON_ADV_UPDATE_MS = 300;

static constexpr float AREA_W = 10.0f;
static constexpr float AREA_H = 10.0f;
static constexpr float CELL_W = 2.0f;
static constexpr float CELL_H = 2.0f;
static constexpr int GRID_NX = 5;
static constexpr int GRID_NY = 5;

static constexpr char ANCHOR_NAMES[4][16] = {
    "ANCHOR_A",
    "ANCHOR_B",
    "ANCHOR_C",
    "ANCHOR_D"};

static constexpr float ANCHOR_X[4] = {2, 8, 8, 2};
static constexpr float ANCHOR_Y[4] = {2, 2, 8, 8};

static constexpr float A_REF[4] = {-59.0f, -59.0f, -59.0f, -59.0f};
static constexpr float N_PLE[4] = {2.5f, 2.5f, 2.5f, 2.5f};

static constexpr size_t NUM_ANCHORS = 4;

static constexpr unsigned long SERIAL_INIT_DELAY_MS = 3000;
static constexpr unsigned long BLE_CONNECTION_RECOVERY_DELAY_MS = 200;
static constexpr unsigned long ERROR_HALT_LOOP_DELAY_MS = 1000;
static constexpr unsigned long REBOOT_DELAY_MS = 5000;

static constexpr unsigned long TIME_SLOT_DURATION_MS = 25;
static constexpr unsigned long TIME_SLOT_OFFSET_MS[4] = {0, 25, 50, 75};

static_assert(TIME_SLOT_DURATION_MS * NUM_ANCHORS == 100,
              "Time slots must equal advertising interval (4 × 25ms = 100ms)");
static_assert(TIME_SLOT_OFFSET_MS[0] == 0 &&
                  TIME_SLOT_OFFSET_MS[1] == TIME_SLOT_DURATION_MS &&
                  TIME_SLOT_OFFSET_MS[2] == TIME_SLOT_DURATION_MS * 2 &&
                  TIME_SLOT_OFFSET_MS[3] == TIME_SLOT_DURATION_MS * 3,
              "Time slot offsets must be sequential (0, 25, 50, 75ms)");

static constexpr uint16_t BLE_SCAN_INTERVAL_MS = 100;
static constexpr uint16_t BLE_SCAN_WINDOW_MS = 99;

static_assert(BLE_SCAN_WINDOW_MS <= BLE_SCAN_INTERVAL_MS,
              "BLE_SCAN_WINDOW_MS must be <= BLE_SCAN_INTERVAL_MS");

static constexpr uint8_t BOOT_SCAN_MAX_RETRIES = 3;
static constexpr unsigned long BOOT_SCAN_RETRY_DELAY_MS = 1000;
static constexpr unsigned long BLE_STACK_SETTLE_DELAY_MS = 1000;

static constexpr unsigned long ERROR_RECOVERY_DELAY_MS = 1000;
static constexpr unsigned long MAIN_LOOP_IDLE_DELAY_MS = 1000;

static constexpr unsigned long MAIN_LOOP_DELAY_MS = 10;
static constexpr unsigned long MEMORY_CHECK_INTERVAL_MS = 30000;

static constexpr unsigned long FACTORY_RESET_BUTTON_HOLD_MS = 3000;
static constexpr unsigned long FACTORY_RESET_REBOOT_DELAY_MS = 1000;

static constexpr unsigned long BLE_PER_ANCHOR_TIMEOUT_MS = 2500;
static constexpr int BLE_MAX_CONNECTION_RETRIES = 2;

static constexpr size_t LOW_MEMORY_THRESHOLD_BYTES = 50000;

static_assert(AREA_W > 0 && AREA_W <= 100, "AREA_W must be between 0-100 meters");
static_assert(AREA_H > 0 && AREA_H <= 100, "AREA_H must be between 0-100 meters");
static_assert(CELL_W > 0 && CELL_W <= AREA_W, "CELL_W must be positive and <= AREA_W");
static_assert(CELL_H > 0 && CELL_H <= AREA_H, "CELL_H must be positive and <= AREA_H");

static_assert(GRID_NX > 0 && GRID_NX <= 26, "GRID_NX must be 1-26 (A-Z columns)");
static_assert(GRID_NY > 0 && GRID_NY <= 26, "GRID_NY must be 1-26 (1-26 rows)");
static_assert(GRID_NX * CELL_W == AREA_W, "Grid width mismatch: GRID_NX * CELL_W != AREA_W");
static_assert(GRID_NY * CELL_H == AREA_H, "Grid height mismatch: GRID_NY * CELL_H != AREA_H");

static_assert(ANCHOR_X[0] >= 0 && ANCHOR_X[0] <= AREA_W, "ANCHOR_A X coordinate out of bounds");
static_assert(ANCHOR_Y[0] >= 0 && ANCHOR_Y[0] <= AREA_H, "ANCHOR_A Y coordinate out of bounds");
static_assert(ANCHOR_X[1] >= 0 && ANCHOR_X[1] <= AREA_W, "ANCHOR_B X coordinate out of bounds");
static_assert(ANCHOR_Y[1] >= 0 && ANCHOR_Y[1] <= AREA_H, "ANCHOR_B Y coordinate out of bounds");
static_assert(ANCHOR_X[2] >= 0 && ANCHOR_X[2] <= AREA_W, "ANCHOR_C X coordinate out of bounds");
static_assert(ANCHOR_Y[2] >= 0 && ANCHOR_Y[2] <= AREA_H, "ANCHOR_C Y coordinate out of bounds");
static_assert(ANCHOR_X[3] >= 0 && ANCHOR_X[3] <= AREA_W, "ANCHOR_D X coordinate out of bounds");
static_assert(ANCHOR_Y[3] >= 0 && ANCHOR_Y[3] <= AREA_H, "ANCHOR_D Y coordinate out of bounds");

static_assert(A_REF[0] >= -100 && A_REF[0] <= 0, "A_REF[0] must be in range -100 to 0 dBm");
static_assert(A_REF[1] >= -100 && A_REF[1] <= 0, "A_REF[1] must be in range -100 to 0 dBm");
static_assert(A_REF[2] >= -100 && A_REF[2] <= 0, "A_REF[2] must be in range -100 to 0 dBm");
static_assert(A_REF[3] >= -100 && A_REF[3] <= 0, "A_REF[3] must be in range -100 to 0 dBm");
static_assert(N_PLE[0] >= 1.5f && N_PLE[0] <= 6.0f, "N_PLE[0] must be in range 1.5 to 6.0");
static_assert(N_PLE[1] >= 1.5f && N_PLE[1] <= 6.0f, "N_PLE[1] must be in range 1.5 to 6.0");
static_assert(N_PLE[2] >= 1.5f && N_PLE[2] <= 6.0f, "N_PLE[2] must be in range 1.5 to 6.0");
static_assert(N_PLE[3] >= 1.5f && N_PLE[3] <= 6.0f, "N_PLE[3] must be in range 1.5 to 6.0");

static_assert(BLE_PER_ANCHOR_TIMEOUT_MS >= 1000 && BLE_PER_ANCHOR_TIMEOUT_MS <= 10000,
              "BLE_PER_ANCHOR_TIMEOUT_MS must be 1-10 seconds");
static_assert(BLE_MAX_CONNECTION_RETRIES >= 1 && BLE_MAX_CONNECTION_RETRIES <= 5,
              "BLE_MAX_CONNECTION_RETRIES must be 1-5");

static_assert(sizeof(ANCHOR_NAMES[0]) <= 32, "Anchor name too long (max 32 chars)");
static_assert(sizeof(ANCHOR_NAMES[1]) <= 32, "Anchor name too long (max 32 chars)");
static_assert(sizeof(ANCHOR_NAMES[2]) <= 32, "Anchor name too long (max 32 chars)");
static_assert(sizeof(ANCHOR_NAMES[3]) <= 32, "Anchor name too long (max 32 chars)");

inline void validateConfiguration(const char *role)
{
    LOG_I(role, "========== CONFIGURATION VALIDATION ==========");

    float dx1 = ANCHOR_X[1] - ANCHOR_X[0];
    float dy1 = ANCHOR_Y[1] - ANCHOR_Y[0];
    float dx2 = ANCHOR_X[2] - ANCHOR_X[0];
    float dy2 = ANCHOR_Y[2] - ANCHOR_Y[0];

    float cross = dx1 * dy2 - dy1 * dx2;
    if (fabs(cross) < 0.1f)
    {
        LOG_W(role, "WARNING: Anchors may be collinear (poor geometry)");
        LOG_W(role, "   This will reduce positioning accuracy!");
        LOG_W(role, "   Recommended: Place anchors in a square/rectangle");
    }
    else
    {
        LOG_I(role, "[OK] Anchor geometry: GOOD (non-collinear)");
    }

    float min_spacing = 1000.0f;
    for (int i = 0; i < 4; i++)
    {
        for (int j = i + 1; j < 4; j++)
        {
            float dx = ANCHOR_X[i] - ANCHOR_X[j];
            float dy = ANCHOR_Y[i] - ANCHOR_Y[j];
            float spacing = sqrt(dx * dx + dy * dy);
            if (spacing < min_spacing)
                min_spacing = spacing;
        }
    }

    if (min_spacing < 2.0f)
    {
        LOG_W(role, "WARNING: Minimum anchor spacing: %.2fm (recommend >2m)", min_spacing);
        LOG_W(role, "   Close anchors reduce accuracy!");
    }
    else
    {
        LOG_I(role, "[OK] Anchor spacing: %.2fm (good)", min_spacing);
    }

    LOG_I(role, "System: RAW RSSI COLLECTION ONLY");

    LOG_I(role, "==========================================");
}
