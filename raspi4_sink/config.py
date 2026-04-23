from typing import Tuple

SERVICE_UUID = "f0e1d2c3-b4a5-9687-7856-453423120001"
CHAR_UUID = "f0e1d2c3-b4a5-9687-7856-453423120002"
CH_CONTROL_UUID = "f0e1d2c3-b4a5-9687-7856-453423120004"
CH_STATS_UUID = "f0e1d2c3-b4a5-9687-7856-453423120005"

CMD_TRIGGER_SCAN = 0x01
CMD_STATUS_QUERY = 0x02

RESP_SCAN_STARTED = 0x11
RESP_SCAN_COMPLETE = 0x12
RESP_SCAN_FAILED = 0x13
RESP_SCAN_IN_PROGRESS = 0x14

ANCHORS = {
    "ANCHOR_A": {
        "mac": "D0:CF:13:15:78:2D",
        "index": 0,
        "position": (2.0, 2.0),
    },
    "ANCHOR_B": {
        "mac": "D0:CF:13:15:9D:CD",
        "index": 1,
        "position": (8.0, 2.0),
    },
    "ANCHOR_C": {
        "mac": "D0:CF:13:15:B0:FD",
        "index": 2,
        "position": (8.0, 8.0),
    },
    "ANCHOR_D": {
        "mac": "D0:CF:13:14:F3:55",
        "index": 3,
        "position": (2.0, 8.0),
    },
}

ANCHOR_NAMES = list(ANCHORS.keys())
ANCHOR_POSITIONS = [info["position"] for info in ANCHORS.values()]

BEACON_MAC = "FC:01:2C:D1:39:05"
MFG_ID = 0xFFFF
ADV_PAYLOAD_LEN = 9

AREA_WIDTH = 10.0
AREA_HEIGHT = 10.0
CELL_WIDTH = 2.0
CELL_HEIGHT = 2.0
GRID_COLS = 5
GRID_ROWS = 5

SCAN_INTERVAL_SEC = 22
SCAN_TIMEOUT_SEC = 10
CONNECTION_TIMEOUT_SEC = 12

MAX_RETRIES = 2
RETRY_DELAY_SEC = 3

PARALLEL_CONNECTION_TIMEOUT_SEC = 8

PARALLEL_STAGGER_DELAY_MS = 150

ONDEMAND_SCAN_TRIGGER_TIMEOUT_SEC = 4
ONDEMAND_SCAN_DURATION_SEC = 3
ONDEMAND_SCAN_BUFFER_SEC = 4

PARALLEL_MAX_RETRIES = 3
PARALLEL_RETRY_DELAY_SEC = 1

PARALLEL_FAILURE_THRESHOLD = 3

BLE_DISCOVER_TIMEOUT_SEC = 5.0
BLE_SCAN_LOG_VERBOSE = True

DEFAULT_SAMPLES_PER_CELL = 100

MISSING_RSSI_VALUE = -999

RSSI_MIN = -100
RSSI_MAX = 0

RSSI_NO_BEACON = -128

MAX_DATA_AGE_MS = 15000

CSV_FILENAME_FORMAT = "{ground_truth_cell}_{timestamp}.csv"

BOUNDARY_JSON_FILENAME_FORMAT = "boundary_{ground_truth_cell}_{timestamp}.json"

CSV_HEADER = [
    "timestamp",
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_cell",
    "rssi_A",
    "rssi_B",
    "rssi_C",
    "rssi_D",
    "anchors_valid",
    "sample_number"
]

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

VERBOSE = True
PROGRESS_INTERVAL = 5

MIN_SAMPLES_FOR_CALIBRATION = 10
CALIBRATION_SAMPLES_PER_DISTANCE = 20
CALIBRATION_DEFAULT_DISTANCES = [1.0, 2.0, 4.0, 6.0, 8.0]

HIGH_RSSI_VARIANCE_THRESHOLD_DB = 10
COVARIANCE_CONDITION_NUMBER_THRESHOLD = 1e10
MISSING_DATA_WARNING_THRESHOLD_PERCENT = 20
MISSING_DATA_CRITICAL_THRESHOLD_PERCENT = 50
OVERALL_MISSING_DATA_THRESHOLD_PERCENT = 25

MINIMUM_SAMPLES_PER_CELL = 50

BYTES_PER_KILOBYTE = 1024

NUMBER_OF_ANCHORS = 4
STRUCT_FORMAT_ANCHOR_PAYLOAD = '<HBbIBbHI'
ANCHOR_PAYLOAD_SIZE_BYTES = 16

EXPONENTIAL_BACKOFF_BASE = 2
EXPONENTIAL_BACKOFF_START_EXPONENT = 0

SEPARATOR_WIDTH = 70
DISPLAY_SAMPLES_BATCH = 3


def coordinates_to_grid_cell(x: float, y: float) -> str:
    col = int(x / CELL_WIDTH)
    row = int(y / CELL_HEIGHT)
    
    col = max(0, min(col, GRID_COLS - 1))
    row = max(0, min(row, GRID_ROWS - 1))
    
    cell = chr(ord('A') + col) + str(row + 1)
    return cell


def grid_cell_to_coordinates(cell: str) -> Tuple[float, float]:
    if not cell or len(cell) < 2:
        raise ValueError(f"Invalid cell format: '{cell}' (expected format: A1-E5)")
    
    col_char = cell[0].upper()
    if not ('A' <= col_char <= 'E'):
        raise ValueError(f"Column must be A-E, got: '{col_char}'")
    
    try:
        row_num = int(cell[1])
    except ValueError:
        raise ValueError(f"Row must be numeric 1-5, got: '{cell[1]}'")
    
    if not (1 <= row_num <= 5):
        raise ValueError(f"Row must be 1-5, got: {row_num}")
    
    col = ord(col_char) - ord('A')
    row = row_num - 1
    
    x = (col + 0.5) * CELL_WIDTH
    y = (row + 0.5) * CELL_HEIGHT
    
    return (x, y)


BLE_DISCONNECT_SETTLE_SEC = 0.5

ANCHOR_BOOT_DELAY_SEC = 20

MS_TO_SEC_CONVERSION = 1000.0


if __name__ == "__main__":
    print("=" * 50)
    print("BLE RSSI DATA COLLECTION - CONFIGURATION")
    print("=" * 50)
    print(f"\nBLE Configuration:")
    print(f"  Service UUID: {SERVICE_UUID}")
    print(f"  Char UUID: {CHAR_UUID}")
    print(f"  Anchors: {ANCHOR_NAMES}")
    
    print(f"\nArea Configuration:")
    print(f"  Dimensions: {AREA_WIDTH}m × {AREA_HEIGHT}m")
    print(f"  Grid: {GRID_COLS} × {GRID_ROWS} cells ({CELL_WIDTH}m × {CELL_HEIGHT}m each)")
    print(f"  Total cells: {GRID_COLS * GRID_ROWS}")
    
    print(f"\nData Collection:")
    print(f"  Scan interval: {SCAN_INTERVAL_SEC}s")
    print(f"  Samples per cell: {DEFAULT_SAMPLES_PER_CELL}")
    print(f"  Time per cell: ~{(DEFAULT_SAMPLES_PER_CELL * SCAN_INTERVAL_SEC) / 60:.1f} minutes")
    print(f"  Total collection time: ~{(GRID_COLS * GRID_ROWS * DEFAULT_SAMPLES_PER_CELL * SCAN_INTERVAL_SEC) / 3600:.1f} hours")
    print("=" * 50)
