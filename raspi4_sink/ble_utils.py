#!/usr/bin/env python3

import asyncio
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError

from config import (
    ANCHORS,
    BLE_DISCOVER_TIMEOUT_SEC,
    BLE_SCAN_LOG_VERBOSE,
    RSSI_MIN,
    RSSI_MAX,
    RSSI_NO_BEACON,
    MISSING_RSSI_VALUE,
)

ANCHOR_PAYLOAD_SIZE = 16
ANCHOR_PAYLOAD_FORMAT = "<HBbIBbHI"

ANCHOR_STATS_PAYLOAD_SIZE = 15
ANCHOR_STATS_PAYLOAD_FORMAT = "<HBbbbbbbbBI"


@dataclass
class AnchorPayload:
    company_id: int
    anchor_index: int
    beacon_rssi: int
    timestamp_ms: int
    sync_status: int
    drift_ms: int
    sync_count: int
    last_sync_time_ms: int


def parse_anchor_payload(data: bytes) -> Optional[AnchorPayload]:
    if len(data) < ANCHOR_PAYLOAD_SIZE:
        return None

    try:
        (company_id, anchor_index, beacon_rssi, timestamp_ms,
         sync_status, drift_ms, sync_count, last_sync_time_ms
        ) = struct.unpack(ANCHOR_PAYLOAD_FORMAT, data[:ANCHOR_PAYLOAD_SIZE])

        return AnchorPayload(
            company_id=company_id,
            anchor_index=anchor_index,
            beacon_rssi=beacon_rssi,
            timestamp_ms=timestamp_ms,
            sync_status=sync_status,
            drift_ms=drift_ms,
            sync_count=sync_count,
            last_sync_time_ms=last_sync_time_ms
        )
    except struct.error:
        return None


def parse_anchor_payload_simple(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    payload = parse_anchor_payload(data)
    if payload is None:
        return (None, None)
    return (payload.beacon_rssi, payload.timestamp_ms)


@dataclass
class AnchorStatsPayload:
    company_id: int
    anchor_index: int
    mean: int           # Mean RSSI (dBm)
    median: int         # Median RSSI (dBm)
    std_dev: int        # Std deviation × 10 (e.g., 25 = 2.5 dBm)
    min_rssi: int       # Minimum RSSI (dBm)
    max_rssi: int       # Maximum RSSI (dBm)
    q25: int            # 25th percentile (Q1)
    q75: int            # 75th percentile (Q3)
    outliers: int       # Outlier count (IQR method)
    timestamp_ms: int   # Firmware timestamp when calculated


def parse_anchor_stats_payload(data: bytes) -> Optional[AnchorStatsPayload]:
    if len(data) < ANCHOR_STATS_PAYLOAD_SIZE:
        return None
    
    try:
        fields = struct.unpack(ANCHOR_STATS_PAYLOAD_FORMAT, data[:ANCHOR_STATS_PAYLOAD_SIZE])
        
        mean = fields[2]
        std_dev = fields[4]
        min_rssi = fields[5]
        max_rssi = fields[6]
        
        if mean > 0:
            print(f"[ERROR] Invalid RSSI mean: {mean} (must be <= 0). Hex: {data[:15].hex()}")
            return None
        if std_dev < 0:
            print(f"[ERROR] Invalid std_dev: {std_dev} (must be >= 0). Hex: {data[:15].hex()}")
            return None
        if min_rssi > max_rssi:
            print(f"[ERROR] Invalid range: min={min_rssi} > max={max_rssi}. Hex: {data[:15].hex()}")
            return None
        
        return AnchorStatsPayload(
            company_id=fields[0],
            anchor_index=fields[1],
            mean=fields[2],
            median=fields[3],
            std_dev=fields[4],
            min_rssi=fields[5],
            max_rssi=fields[6],
            q25=fields[7],
            q75=fields[8],
            outliers=fields[9],
            timestamp_ms=fields[10]
        )
    except struct.error:
        return None


async def scan_for_anchors(verbose: bool = BLE_SCAN_LOG_VERBOSE) -> Optional[Dict[str, BLEDevice]]:
    if verbose:
        print("[SCAN] Scanning for anchors (Optimized v2.5)...")
        print("       Using shared ble_utils.scan_for_anchors() with O(N) complexity")

    try:
        target_macs = {name: info['mac'].lower() for name, info in ANCHORS.items()}
        
        mac_to_name = {mac: name for name, mac in target_macs.items()}

        if verbose:
            print(f"       [BLE] Scanning for {len(target_macs)} anchors ({BLE_DISCOVER_TIMEOUT_SEC}s timeout)...")

        devices_with_adv = await BleakScanner.discover(
            timeout=BLE_DISCOVER_TIMEOUT_SEC,
            return_adv=True
        )

        if verbose:
            print(f"   Found {len(devices_with_adv)} BLE devices total")

        anchor_devices = {}
        for address, (device, adv_data) in devices_with_adv.items():
            device_mac = address.lower()
            
            if device_mac in mac_to_name:
                anchor_name = mac_to_name[device_mac]
                anchor_devices[anchor_name] = device
                
                if verbose:
                    rssi_val = adv_data.rssi if adv_data else None
                    rssi_str = f"{rssi_val} dBm" if rssi_val is not None else "RSSI N/A"
                    print(f"       [OK] {anchor_name}: {device_mac} ({rssi_str})")
                
                if len(anchor_devices) == 4:
                    if verbose:
                        remaining = len(devices_with_adv) - len(anchor_devices)
                        print(f"       [EARLY EXIT] All 4 anchors found, skipping {remaining} remaining devices")
                    break

        missing = set(ANCHORS.keys()) - set(anchor_devices.keys())
        if missing:
            error_msg = f"Missing anchors: {', '.join(missing)}"
            if verbose:
                print(f"       [ERROR] {error_msg}")
            return None

        if verbose:
            print(f"       [OK] Scan complete: {len(anchor_devices)}/4 anchors discovered")
            print()

        return anchor_devices

    except BleakError as e:
        if verbose:
            print(f"       [ERROR] BLE scan failed: {e}")
        return None
    except asyncio.TimeoutError:
        if verbose:
            print(f"       [ERROR] Scan timeout after {BLE_DISCOVER_TIMEOUT_SEC}s")
        return None
    except Exception as e:
        if verbose:
            print(f"       [ERROR] Unexpected scan error: {e}")
        return None


def validate_ble_device(device: BLEDevice, expected_name: Optional[str] = None) -> bool:
    if device is None:
        return False

    if not hasattr(device, 'address') or not device.address:
        return False

    if expected_name and device.name != expected_name:
        return False

    return True


def format_ble_error(error: Exception) -> str:
    error_str = str(error).lower()

    # Common BlueZ/Bleak errors with user-friendly messages (case-insensitive)
    if "operation already in progress" in error_str:
        return "BLE adapter busy (another operation in progress). Retry in a few seconds."

    if "device with address" in error_str and "was not found" in error_str:
        return "Anchor not found. Ensure device is powered on and advertising."

    if "bluetooth adapter" in error_str and "not found" in error_str:
        return "Bluetooth adapter not available. Check if Bluetooth is enabled."

    if "dbus.error" in error_str:
        return f"BlueZ D-Bus error: {error_str}"

    if "timeout" in error_str:
        return "Connection timeout. Anchor may be out of range or not responding."

    return f"BLE error: {error_str}"


def is_valid_rssi(rssi: int) -> bool:
    return RSSI_MIN <= rssi <= RSSI_MAX


def is_valid_measured_rssi(rssi: int) -> bool:
    if rssi == RSSI_NO_BEACON or rssi == MISSING_RSSI_VALUE:
        return False
    return RSSI_MIN <= rssi <= RSSI_MAX


def is_beacon_detected(rssi: int) -> bool:
    return rssi != RSSI_NO_BEACON and rssi != MISSING_RSSI_VALUE


__all__ = [
    'ANCHOR_PAYLOAD_SIZE',
    'ANCHOR_PAYLOAD_FORMAT',
    'AnchorPayload',
    'parse_anchor_payload',
    'parse_anchor_payload_simple',
    'AnchorStatsPayload',
    'parse_anchor_stats_payload',
    'scan_for_anchors',
    'validate_ble_device',
    'format_ble_error',
    'is_valid_rssi',
    'is_valid_measured_rssi',
    'is_beacon_detected',
]
