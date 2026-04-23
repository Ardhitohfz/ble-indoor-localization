#!/usr/bin/env python3

import asyncio
import csv
import signal
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError
from ble_utils import scan_for_anchors, format_ble_error, is_valid_rssi, parse_anchor_stats_payload
from config import (
    SERVICE_UUID,
    CHAR_UUID,
    CH_CONTROL_UUID,
    CH_STATS_UUID,
    ANCHORS,
    ANCHOR_NAMES,
    PARALLEL_CONNECTION_TIMEOUT_SEC,
    PARALLEL_STAGGER_DELAY_MS,
    PARALLEL_MAX_RETRIES,
    PARALLEL_RETRY_DELAY_SEC,
    PARALLEL_FAILURE_THRESHOLD,
    CMD_TRIGGER_SCAN,
    ONDEMAND_SCAN_TRIGGER_TIMEOUT_SEC,
    ONDEMAND_SCAN_DURATION_SEC,
    ONDEMAND_SCAN_BUFFER_SEC,
    RSSI_NO_BEACON,
    MISSING_RSSI_VALUE,
    BLE_DISCONNECT_SETTLE_SEC,
    ANCHOR_BOOT_DELAY_SEC,
    MS_TO_SEC_CONVERSION,
)

class ParallelDataCollector:

    def __init__(
        self,
        output_file: Optional[str] = None,
        verbose: bool = True,
        ground_truth_x: Optional[float] = None,
        ground_truth_y: Optional[float] = None,
        ground_truth_cell: Optional[str] = None,
        beacon_moving: bool = False,
        environment: str = "basement_parking",
        notes: str = ""
    ):
        self.verbose = verbose
        self.running = True
        self.samples_collected = 0
        self.failed_samples = 0
        self.consecutive_failures = 0

        self.ground_truth_x = ground_truth_x
        self.ground_truth_y = ground_truth_y
        self.ground_truth_cell = ground_truth_cell
        self.beacon_moving = beacon_moving
        self.environment = environment
        self.notes = notes

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"dataset_{timestamp}.csv"

        output_path = Path(output_file)
        if not output_path.is_absolute():
            script_dir = Path(__file__).parent
            dataset_dir = script_dir / "data" / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            self.output_file = dataset_dir / output_path.name
        else:
            self.output_file = output_path

        self.csv_writer: Any = None
        self.csv_file: Any = None

        self.timing_stats = []
        self.success_rates = []
        self.anchor_devices: Dict[str, BLEDevice] = {}
        self.scan_refresh_counter = 0
        self.last_payload: Dict[str, Tuple[int, int]] = {}
        
        self.failed_anchors = {name: 0 for name in ["ANCHOR_A", "ANCHOR_B", "ANCHOR_C", "ANCHOR_D"]}
        self.successful_reads = {name: 0 for name in ["ANCHOR_A", "ANCHOR_B", "ANCHOR_C", "ANCHOR_D"]}
        
        self.last_stats: Dict[str, Optional[Dict]] = {}

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.log(f"[FILE] Output file: {self.output_file}")

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log message with timestamp.

        Args:
            message: Message to log
            level: Log level (INFO, WARN, ERROR)
        """
        if not self.verbose and level == "INFO":
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "[INFO]",
            "WARN": "[WARN]",
            "ERROR": "[ERROR]"
        }.get(level, "•")

        print(f"[{timestamp}] {prefix} {message}")

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        sig_name = signal.Signals(signum).name
        self.log(f"Received {sig_name}, shutting down gracefully...", "WARN")
        self.running = False

    async def trigger_anchor_scan(
        self,
        anchor_name: str,
        ble_device: BLEDevice
    ) -> bool:
        
        try:
            # Brief connection to trigger scan
            async with BleakClient(
                ble_device,
                timeout=PARALLEL_CONNECTION_TIMEOUT_SEC
            ) as client:
                # Send trigger command
                await asyncio.wait_for(
                    client.write_gatt_char(
                        CH_CONTROL_UUID,
                        bytes([CMD_TRIGGER_SCAN]),
                        response=True
                    ),
                    timeout=ONDEMAND_SCAN_TRIGGER_TIMEOUT_SEC
                )
                self.log(f"{anchor_name}: Scan triggered", "DEBUG")

            await asyncio.sleep(BLE_DISCONNECT_SETTLE_SEC)

            return True

        except Exception as e:
            self.log(f"{anchor_name}: Failed to trigger scan: {e}", "WARN")
            return False

    async def read_single_anchor(
        self,
        anchor_name: str,
        ble_device: BLEDevice,
        stagger_delay_ms: int = 0,
        retry_count: int = 0
    ) -> Tuple[Optional[int], float, Optional[Dict[str, Any]]]:
        if stagger_delay_ms > 0:
            await asyncio.sleep(stagger_delay_ms / MS_TO_SEC_CONVERSION)

        start_time = time.time()

        async def _read_payload(client) -> Tuple[Optional[int], Optional[int], Optional[Dict]]:
            data = await client.read_gatt_char(CHAR_UUID)
            if len(data) < 16:
                self.log(
                    f"{anchor_name}: Invalid data length {len(data)} bytes",
                    "WARN"
                )
                return None, None, None
            try:
                company_id, anchor_idx, beacon_rssi, ts_ms, sync_status, drift_ms, sync_count, last_sync = struct.unpack(
                    "<HBbIBbHI", data[:16])
                
                # Read statistical features (v6.0 - Always enabled)
                stats_dict = None
                try:
                    stats_data = await client.read_gatt_char(CH_STATS_UUID)
                    stats_payload = parse_anchor_stats_payload(stats_data)
                    if stats_payload:
                        stats_dict = {
                            'mean': stats_payload.mean,
                            'median': stats_payload.median,
                            'std': stats_payload.std_dev / 10.0,  # Convert back from ×10
                            'min': stats_payload.min_rssi,
                            'max': stats_payload.max_rssi,
                            'q25': stats_payload.q25,
                            'q75': stats_payload.q75,
                            'outliers': stats_payload.outliers
                        }
                except Exception as e:
                    # Stats characteristic not available (firmware v5.x or earlier)
                    self.log(f"{anchor_name}: Stats not available: {e!r}", "DEBUG")
                
                return beacon_rssi, ts_ms, stats_dict
            except struct.error as e:
                self.log(f"{anchor_name}: Parse error: {e!r}", "ERROR")
                return None, None, None

        try:
            # Connection phase (v1.3: Using BLEDevice object - CRITICAL FIX!)
            # NOTE: Scan already triggered separately before this function is called
            async with BleakClient(
                ble_device,  # BLEDevice object, NOT raw MAC address!
                timeout=PARALLEL_CONNECTION_TIMEOUT_SEC
            ) as client:
                beacon_rssi, ts_ms, stats_dict = await _read_payload(client)

                if beacon_rssi is None:
                    self.log(f"{anchor_name}: No data received", "WARN")
                    return None, 0.0, None

                if beacon_rssi == RSSI_NO_BEACON:
                    self.log(f"{anchor_name}: NO_BEACON ({beacon_rssi} dBm)", "WARN")
                    return MISSING_RSSI_VALUE, time.time() - start_time, None

                if not is_valid_rssi(beacon_rssi):
                    self.log(f"{anchor_name}: Invalid RSSI {beacon_rssi} dBm", "WARN")
                    return MISSING_RSSI_VALUE, time.time() - start_time, None

                if ts_ms is not None:
                    self.last_payload[anchor_name] = (beacon_rssi, ts_ms)
                if stats_dict is not None:
                    self.last_stats[anchor_name] = stats_dict

                return beacon_rssi, time.time() - start_time, stats_dict

        except asyncio.TimeoutError:
            if retry_count < PARALLEL_MAX_RETRIES - 1:
                self.log(f"{anchor_name}: Timeout, retrying...", "WARN")
            else:
                self.log(f"{anchor_name}: Timeout after {retry_count + 1} attempts", "ERROR")
            return None, 0.0, None

        except BleakError as e:
            self.log(f"{anchor_name}: {format_ble_error(e)} ({type(e).__name__}: {e!r})", "ERROR")
            return None, 0.0, None

        except Exception as e:
            self.log(f"{anchor_name}: Unexpected error: {e!r}", "ERROR")
            return None, 0.0, None

    async def _get_anchor_devices(self, force_rescan: bool = False) -> Dict[str, BLEDevice]:
        if self.anchor_devices and not force_rescan:
            return self.anchor_devices

        anchor_devices = await scan_for_anchors(verbose=self.verbose)
        if anchor_devices is None:
            return {}
        self.anchor_devices = anchor_devices
        self.scan_refresh_counter = 0
        return anchor_devices

    async def _cleanup_clients(self, clients: Dict[str, BleakClient]) -> None:
        disconnect_tasks = []
        for client in clients.values():
            if client.is_connected:
                disconnect_tasks.append(client.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

    async def _phase1_connect(
        self, 
        anchor_devices: Dict[str, BLEDevice]
    ) -> Tuple[Optional[Dict[str, BleakClient]], bool]:
        if self.verbose:
            self.log("[CONN] Phase 1/5: Connecting to all anchors...", "INFO")

        clients = {}
        connect_tasks = []
        anchor_names = ANCHOR_NAMES

        for name in anchor_names:
            client = BleakClient(anchor_devices[name], timeout=PARALLEL_CONNECTION_TIMEOUT_SEC)
            clients[name] = client
            task = asyncio.create_task(client.connect())
            task.set_name(name)
            connect_tasks.append(task)
        # Benefit: 75% faster error detection (2s vs 8s on timeout failure)
        done, pending = await asyncio.wait(
            connect_tasks,
            return_when=asyncio.FIRST_EXCEPTION
        )

        # Cancel any still-running tasks (fast-fail optimization)
        if pending:
            for task in pending:
                task.cancel()
                if self.verbose:
                    self.log(f"[CANCEL] {task.get_name()}: Connection cancelled (fast-fail)", "WARN")
            
            # Wait for cancellations to complete gracefully
            await asyncio.gather(*pending, return_exceptions=True)

        # Check results from completed tasks
        all_connected = True
        for task in done:
            anchor_name = task.get_name()
            try:
                task.result()  # Raises exception if task failed
                if not clients[anchor_name].is_connected:
                    self.log(f"[ERROR] {anchor_name}: Connection not established", "ERROR")
                    all_connected = False
            except Exception as e:
                self.log(f"[ERROR] {anchor_name}: Connect failed: {e}", "ERROR")
                all_connected = False

        if not all_connected:
            await self._cleanup_clients(clients)
            return None, False

        if self.verbose:
            self.log("[OK] All anchors connected (connections staying open)", "INFO")
        
        return clients, True

    async def _phase2_trigger(self, clients: Dict[str, BleakClient]) -> bool:
        if self.verbose:
            self.log("[PHASE 2/5] Sending scan trigger commands...", "INFO")

        anchor_names = ANCHOR_NAMES

        # Clean Code v2.3: Define helper function ONCE outside loop (KISS principle)
        # This avoids redefining the function on each iteration
        async def trigger_single_anchor(anchor_name: str, client: BleakClient) -> Tuple[str, bool]:
            """Trigger scan on a single anchor. Returns (name, success)."""
            try:
                await client.write_gatt_char(CH_CONTROL_UUID, bytes([CMD_TRIGGER_SCAN]))
                return (anchor_name, True)
            except Exception as e:
                self.log(f"[WARN] {anchor_name}: Trigger failed: {e}", "WARN")
                return (anchor_name, False)

        # Create tasks using the helper function
        trigger_tasks = [
            trigger_single_anchor(name, clients[name])
            for name in anchor_names
        ]

        trigger_results = await asyncio.gather(*trigger_tasks, return_exceptions=True)

        trigger_success_count = sum(1 for result in trigger_results 
                                   if isinstance(result, tuple) and len(result) == 2 and result[1])
        
        if trigger_success_count < 4:
            self.log(f"[WARN] Only {trigger_success_count}/4 anchors accepted trigger", "WARN")
            return False

        if self.verbose:
            self.log("[OK] All triggers sent successfully", "INFO")

        return True

    async def _phase4_read(
        self,
        clients: Dict[str, BleakClient]
    ) -> Tuple[Dict[str, int], bool]:
        if self.verbose:
            self.log("[PHASE 4/5] Reading RSSI data...", "INFO")

        anchor_names = ANCHOR_NAMES

        async def read_single_anchor(anchor_name: str, client: BleakClient) -> Tuple[str, Optional[int]]:
            try:
                data = await client.read_gatt_char(CHAR_UUID)
                if len(data) < 16:
                    self.log(f"[WARN] {anchor_name}: Invalid data length: {len(data)}", "WARN")
                    return (anchor_name, None)

                try:
                    _, _, rssi, _, _, _, _, _ = struct.unpack("<HBbIBbHI", data[:16])
                except struct.error as e:
                    self.log(f"[ERROR] {anchor_name}: Parse error: {e}", "ERROR")
                    return (anchor_name, None)

                # Read statistical features (v6.0 - Always enabled)
                try:
                    stats_data = await client.read_gatt_char(CH_STATS_UUID)
                    # CRITICAL DEBUG: Verify actual BLE data size
                    self.log(f"[DEBUG] {anchor_name}: stats_data size = {len(stats_data)} bytes, hex = {stats_data.hex()[:40]}...", "DEBUG")
                    stats_payload = parse_anchor_stats_payload(stats_data)
                    if stats_payload:
                        self.last_stats[anchor_name] = {
                            'mean': stats_payload.mean,
                            'median': stats_payload.median,
                            'std': stats_payload.std_dev / 10.0,  # Convert back from ×10
                            'min': stats_payload.min_rssi,
                            'max': stats_payload.max_rssi,
                            'q25': stats_payload.q25,
                            'q75': stats_payload.q75,
                            'outliers': stats_payload.outliers
                        }
                except Exception as e:
                    # Stats characteristic not available (firmware v5.x or earlier)
                    if self.verbose:
                        self.log(f"{anchor_name}: Stats not available: {e!r}", "DEBUG")

                if is_valid_rssi(rssi):
                    return (anchor_name, rssi)
                else:
                    self.log(f"[WARN] {anchor_name}: Invalid RSSI: {rssi} dBm", "WARN")
                    return (anchor_name, None)

            except Exception as e:
                self.log(f"[ERROR] {anchor_name}: Read failed: {e}", "ERROR")
                return (anchor_name, None)

        read_tasks = [
            read_single_anchor(name, clients[name])
            for name in anchor_names
        ]

        read_results = await asyncio.gather(*read_tasks, return_exceptions=True)

        rssi_values = {}
        all_success = True

        for result in read_results:
            if not isinstance(result, tuple) or len(result) != 2:
                all_success = False
                continue
            name, rssi = result
            if rssi is None:
                all_success = False
                self.failed_anchors[name] += 1
                self.log(f"[ERROR] {name}: Failed to get RSSI", "ERROR")
            else:
                rssi_values[name] = rssi
                self.successful_reads[name] += 1

        return rssi_values, all_success

    async def collect_parallel_sample(self) -> Optional[Dict[str, int]]:
        for attempt in range(1, PARALLEL_MAX_RETRIES + 1):
            force_rescan = attempt > 1
            anchor_devices = await self._get_anchor_devices(force_rescan=force_rescan)

            if not anchor_devices:
                self.log("Failed to discover anchors", "ERROR")
                if attempt < PARALLEL_MAX_RETRIES:
                    await asyncio.sleep(PARALLEL_RETRY_DELAY_SEC)
                    self.log(f"Retry attempt {attempt + 1}/{PARALLEL_MAX_RETRIES}...", "WARN")
                continue

            try:
                clients, connected = await self._phase1_connect(anchor_devices)
                if not connected or clients is None:
                    if attempt < PARALLEL_MAX_RETRIES:
                        await asyncio.sleep(PARALLEL_RETRY_DELAY_SEC)
                        self.log(f"Retry attempt {attempt + 1}/{PARALLEL_MAX_RETRIES}...", "WARN")
                    continue

                triggered = await self._phase2_trigger(clients)
                if not triggered:
                    await self._cleanup_clients(clients)
                    if attempt < PARALLEL_MAX_RETRIES:
                        await asyncio.sleep(PARALLEL_RETRY_DELAY_SEC)
                        self.log(f"Retry attempt {attempt + 1}/{PARALLEL_MAX_RETRIES}...", "WARN")
                    continue

                wait_time = ONDEMAND_SCAN_DURATION_SEC + ONDEMAND_SCAN_BUFFER_SEC
                if self.verbose:
                    self.log(f"[WAIT] Phase 3/5: Waiting {wait_time}s for scans (staying connected)...", "INFO")
                await asyncio.sleep(wait_time)

                # PHASE 4: Read RSSI data in parallel
                rssi_values, all_success = await self._phase4_read(clients)

                # PHASE 5: Disconnect all anchors
                if self.verbose:
                    self.log("[CONN] Phase 5/5: Disconnecting all anchors...", "INFO")
                await self._cleanup_clients(clients)

                # Check if we got all 4 anchors
                if all_success and len(rssi_values) == 4:
                    if self.verbose:
                        self.log("[OK] Sample collected successfully from all 4 anchors", "INFO")
                    return rssi_values

                # Partial failure, retry
                if attempt < PARALLEL_MAX_RETRIES:
                    await asyncio.sleep(PARALLEL_RETRY_DELAY_SEC)
                    self.log(f"Retry attempt {attempt + 1}/{PARALLEL_MAX_RETRIES}...", "WARN")

            except Exception as e:
                self.log(f"[ERROR] Unexpected error in sample collection: {e}", "ERROR")
                
                # Clean up connections on exception
                if 'clients' in locals() and clients is not None:
                    await self._cleanup_clients(clients)
                
                if attempt < PARALLEL_MAX_RETRIES:
                    await asyncio.sleep(PARALLEL_RETRY_DELAY_SEC)
                    self.log(f"Retry attempt {attempt + 1}/{PARALLEL_MAX_RETRIES}...", "WARN")

        # All retries exhausted
        self.log("[ERROR] Failed to collect sample with all anchors after 3 attempts", "ERROR")
        return None

    def init_csv_writer(self) -> None:
        try:
            self.csv_file = open(self.output_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            header = [
                'timestamp',
                'ground_truth_x',
                'ground_truth_y',
                'ground_truth_cell',
                'rssi_A',
                'rssi_B',
                'rssi_C',
                'rssi_D',
            ]
            
            stats_headers = [
                'rssiA_mean', 'rssiA_median', 'rssiA_std', 'rssiA_min', 'rssiA_max',
                'rssiA_q25', 'rssiA_q75', 'rssiA_outliers',
                'rssiB_mean', 'rssiB_median', 'rssiB_std', 'rssiB_min', 'rssiB_max',
                'rssiB_q25', 'rssiB_q75', 'rssiB_outliers',
                'rssiC_mean', 'rssiC_median', 'rssiC_std', 'rssiC_min', 'rssiC_max',
                'rssiC_q25', 'rssiC_q75', 'rssiC_outliers',
                'rssiD_mean', 'rssiD_median', 'rssiD_std', 'rssiD_min', 'rssiD_max',
                'rssiD_q25', 'rssiD_q75', 'rssiD_outliers',
            ]
            header.extend(stats_headers)
            
            header.extend([
                'anchors_valid',
                'sample_number',
                'collection_time_ms',
                'beacon_moving',
                'environment',
                'notes'
            ])
            
            self.csv_writer.writerow(header)
            self.csv_file.flush()

            self.log(f"[OK] CSV file initialized: {self.output_file}")
            if self.ground_truth_x is not None and self.ground_truth_y is not None:
                self.log(f"[POS] Ground truth: ({self.ground_truth_x}, {self.ground_truth_y})")
            if self.ground_truth_cell:
                self.log(f"[CELL] Cell: {self.ground_truth_cell}")
            self.log(f"[ENV] Environment: {self.environment}")

        except IOError as e:
            self.log(f"Failed to create CSV file: {e}", "ERROR")
            raise

    def write_sample_to_csv(
        self,
        rssi_values: Dict[str, int],
        sample_number: int,
        collection_time: float
    ):
        timestamp = datetime.now().isoformat()

        valid_count = sum(
            1 for rssi in rssi_values.values()
            if rssi != MISSING_RSSI_VALUE and is_valid_rssi(rssi)
        )

        collection_time_ms = int(collection_time * 1000)

        row = [
            timestamp,
            self.ground_truth_x if self.ground_truth_x is not None else "",
            self.ground_truth_y if self.ground_truth_y is not None else "",
            self.ground_truth_cell if self.ground_truth_cell else "",
            rssi_values.get('ANCHOR_A', MISSING_RSSI_VALUE),
            rssi_values.get('ANCHOR_B', MISSING_RSSI_VALUE),
            rssi_values.get('ANCHOR_C', MISSING_RSSI_VALUE),
            rssi_values.get('ANCHOR_D', MISSING_RSSI_VALUE),
        ]
        
        for anchor_name in ['ANCHOR_A', 'ANCHOR_B', 'ANCHOR_C', 'ANCHOR_D']:
            stats = self.last_stats.get(anchor_name)
            if stats:
                row.extend([
                    stats['mean'],
                    stats['median'],
                    stats['std'],
                    stats['min'],
                    stats['max'],
                    stats['q25'],
                    stats['q75'],
                    stats['outliers']
                ])
            else:
                row.extend(['', '', '', '', '', '', '', ''])
        
        row.extend([
            valid_count,
            sample_number,
            collection_time_ms,
            "true" if self.beacon_moving else "false",
            self.environment,
            self.notes
        ])

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    async def collect_data(self, num_samples: int = 100, interval_sec: float = 1.0) -> None:
     
        self.log("=" * 70)
        self.log("PARALLEL MODE DATA COLLECTION")
        self.log("=" * 70)
        self.log(f"Target samples: {num_samples}")
        self.log(f"Sample interval: {interval_sec}s")
        self.log(f"Estimated time: ~{(num_samples * interval_sec) / 60:.1f} minutes")
        self.log("=" * 70)
        self.log("")

        # CRITICAL FIX v1.4.1: Wait for anchors to complete boot scan
        # Anchors perform blocking boot scan (5-15s) during startup
        # If we trigger scan too early, we get stale boot scan results
        self.log(f"[WAIT] Waiting for anchors to complete boot sequence ({ANCHOR_BOOT_DELAY_SEC}s)...", "INFO")
        self.log("   (Anchors perform beacon scan during boot)", "INFO")
        await asyncio.sleep(ANCHOR_BOOT_DELAY_SEC)  # Clean Code v2.2: Named constant
        self.log("[OK] Anchors ready for on-demand scanning", "INFO")
        self.log("")

        # Initialize CSV
        self.init_csv_writer()

        # Collection loop
        for sample_num in range(1, num_samples + 1):
            if not self.running:
                self.log("Collection stopped by user", "WARN")
                break

            self.log(f"Collecting sample {sample_num}/{num_samples}...")

            # Collect sample
            start_time = time.time()
            rssi_values = await self.collect_parallel_sample()
            collection_time = time.time() - start_time

            if rssi_values is not None:
                # Write to CSV
                self.write_sample_to_csv(rssi_values, sample_num, collection_time)
                self.samples_collected += 1
                self.consecutive_failures = 0

                # BUG-001 FIX v1.4.2: Populate timing_stats and success_rates for progress reporting
                # These lists track collection performance over time for user feedback
                self.timing_stats.append(collection_time)
                self.success_rates.append(1.0)  # 100% success for this sample

                # Progress report
                if sample_num % 10 == 0 and len(self.timing_stats) > 0:
                    avg_time = sum(self.timing_stats) / len(self.timing_stats)
                    avg_success = sum(self.success_rates) / len(self.success_rates) * 100
                    self.log(
                        f"Progress: {sample_num}/{num_samples} | "
                        f"Avg time: {avg_time:.2f}s | "
                        f"Avg success: {avg_success:.1f}%"
                    )

            else:
                # Sample collection failed
                self.failed_samples += 1
                self.consecutive_failures += 1
                # BUG-001 FIX v1.4.2: Track failure in success_rates for accurate statistics
                self.success_rates.append(0.0)  # 0% success for failed sample
                self.log(f"Failed to collect sample {sample_num}", "ERROR")

                # Check failure threshold
                if self.consecutive_failures >= PARALLEL_FAILURE_THRESHOLD:
                    self.log(
                        f"{self.consecutive_failures} consecutive failures - "
                        "check anchor connectivity!",
                        "ERROR"
                    )
                    # In production, could fallback to sequential mode here

            # Wait interval before next sample
            if sample_num < num_samples and self.running:
                await asyncio.sleep(interval_sec)

        # Final statistics
        self.print_final_statistics()

    def print_final_statistics(self) -> None:

        self.log("")
        self.log("=" * 70)
        self.log("DATA COLLECTION COMPLETE")
        self.log("=" * 70)
        self.log(f"Samples collected: {self.samples_collected}")
        self.log(f"Samples failed: {self.failed_samples}")

        if len(self.timing_stats) > 0:
            avg_time = sum(self.timing_stats) / len(self.timing_stats)
            min_time = min(self.timing_stats)
            max_time = max(self.timing_stats)
            self.log(f"Avg collection time: {avg_time:.2f}s")
            self.log(f"Min/Max time: {min_time:.2f}s / {max_time:.2f}s")

        if len(self.success_rates) > 0:
            avg_success = sum(self.success_rates) / len(self.success_rates) * 100
            self.log(f"Avg anchor success rate: {avg_success:.1f}%")

        self.log(f"Output file: {self.output_file}")
        self.log(f"File size: {self.output_file.stat().st_size / 1024:.2f} KB")
        self.log("=" * 70)

    def cleanup(self) -> None:

        if self.csv_file:
            self.csv_file.close()
            self.log("CSV file closed")


def get_float_input(prompt: str, allow_empty: bool = False) -> Optional[float]:

    while True:
        try:
            value = input(f"   {prompt}: ").strip()

            if not value and allow_empty:
                return None

            if not value:
                print("   [ERROR] Value required. Please enter a number.")
                continue

            return float(value)

        except ValueError:
            print("   [ERROR] Invalid number. Please try again.")
        except KeyboardInterrupt:
            print("\n\n[WARN] Input cancelled by user")
            sys.exit(130)


def get_string_input(prompt: str, allow_empty: bool = False) -> str:
    while True:
        try:
            value = input(f"   {prompt}: ").strip()

            if not value and not allow_empty:
                print("   [ERROR] Value required. Please enter text.")
                continue

            return value

        except KeyboardInterrupt:
            print("\n\n[WARN] Input cancelled by user")
            sys.exit(130)


def get_yes_no_input(prompt: str, default: bool = False) -> bool:
    default_str = "Y/n" if default else "y/N"

    while True:
        try:
            value = input(f"   {prompt} ({default_str}): ").strip().lower()

            if not value:
                return default

            if value in ['y', 'yes']:
                return True
            elif value in ['n', 'no']:
                return False
            else:
                print("   [ERROR] Please enter 'y' or 'n'")

        except KeyboardInterrupt:
            print("\n\n[WARN] Input cancelled by user")
            sys.exit(130)


def get_environment_input() -> str:

    environments = [
        ("empty_room", "Empty room (baseline)"),
        ("occupied_room", "Occupied room (realistic)"),
        ("crowded", "Crowded area (stress test)"),
        ("outdoor_clear", "Outdoor clear (minimal obstacles)"),
        ("outdoor_obstructed", "Outdoor obstructed (many obstacles)"),
        ("custom", "Custom (enter manually)")
    ]

    print("\n[ENV] Environment Type:")
    for idx, (env_type, description) in enumerate(environments, 1):
        print(f"   {idx}. {description}")

    while True:
        try:
            choice = input(f"   Select [1-{len(environments)}]: ").strip()

            if not choice:
                print("   [ERROR] Selection required.")
                continue

            try:
                choice_idx = int(choice)
                if 1 <= choice_idx <= len(environments):
                    env_type, _ = environments[choice_idx - 1]

                    if env_type == "custom":
                        return get_string_input("Custom environment name")
                    else:
                        return env_type
                else:
                    print(f"   [ERROR] Please enter a number between 1 and {len(environments)}")
            except ValueError:
                print("   [ERROR] Please enter a valid number")

        except KeyboardInterrupt:
            print("\n\n[WARN] Input cancelled by user")
            sys.exit(130)


def print_interactive_banner():
    print()
    print("=" * 70)
    print("BLE LOCALIZER - INTERACTIVE DATA COLLECTION (Indoor/Outdoor Capable)")
    print("=" * 70)
    print()


def print_configuration_summary(
    ground_truth_x: Optional[float],
    ground_truth_y: Optional[float],
    cell: Optional[str],
    beacon_moving: bool,
    environment: str,
    notes: str,
    samples: int,
    interval: float
):
    print()
    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)

    if ground_truth_x is not None and ground_truth_y is not None:
        print(f"[POS] Ground truth: ({ground_truth_x}, {ground_truth_y})")
    else:
        print("[POS] Ground truth: Not specified")

    if cell:
        print(f"[CELL] Cell/Location: {cell}")
    else:
        print("[CELL] Cell/Location: Not specified")

    print(f"[ENV] Environment: {environment}")
    print(f"[MOVE] Beacon moving: {'Yes' if beacon_moving else 'No'}")

    if notes:
        print(f"[NOTE] Notes: {notes}")

    print(f"[COUNT] Samples: {samples}")
    print(f"[TIME] Interval: {interval}s")
    print(f"[TIME] Estimated time: ~{(samples * (interval + 2)) / 60:.1f} minutes")
    print("=" * 70)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="BLE Localizer - Parallel Mode Data Collection (Multi-Purpose: Indoor/Outdoor)",
        epilog="""
  # Interactive mode (guided prompts)
  python3 rpi_sink_parallel.py --samples 100 --interactive
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of samples to collect (default: 100)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Interval between samples in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV filename (default: auto-generated)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable verbose logging'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode with guided prompts (user-friendly)'
    )

    parser.add_argument(
        '--ground-truth-x',
        type=float,
        default=None,
        help='Ground truth X coordinate in meters (for supervised ML)'
    )
    parser.add_argument(
        '--ground-truth-y',
        type=float,
        default=None,
        help='Ground truth Y coordinate in meters (for supervised ML)'
    )
    parser.add_argument(
        '--cell',
        type=str,
        default=None,
        help='Grid cell label (e.g., A1, B2, C3, D4, E5)'
    )
    parser.add_argument(
        '--beacon-moving',
        action='store_true',
        help='Mark beacon as moving (default: false/stationary)'
    )
    parser.add_argument(
        '--environment',
        type=str,
        default='basement_parking',
        help='Environment type (default: basement_parking with vehicles and concrete columns)'
    )
    parser.add_argument(
        '--notes',
        type=str,
        default='',
        help='Additional notes/metadata for this collection'
    )

    args = parser.parse_args()

    needs_interactive = (
        args.interactive or
        (args.ground_truth_x is None and args.ground_truth_y is None and not args.quiet)
    )

    if needs_interactive:
        # Show interactive banner
        print_interactive_banner()

        print("[POS] Ground Truth Position:")
        # Get ground truth coordinates (if not provided via CLI)
        if args.ground_truth_x is None:
            ground_truth_x = get_float_input(
                "X coordinate (meters), or press Enter to skip",
                allow_empty=True
            )
        else:
            ground_truth_x = args.ground_truth_x
            print(f"   X coordinate (meters): {ground_truth_x} (from CLI)")

        if args.ground_truth_y is None:
            ground_truth_y = get_float_input(
                "Y coordinate (meters), or press Enter to skip",
                allow_empty=True
            )
        else:
            ground_truth_y = args.ground_truth_y
            print(f"   Y coordinate (meters): {ground_truth_y} (from CLI)")

        print("\n[CELL] Location/Cell Label:")
        # Get cell/location label (if not provided via CLI)
        if args.cell is None:
            cell = get_string_input(
                "Cell/Location name (e.g., A1, waypoint_1, corner_NE), or press Enter to skip",
                allow_empty=True
            )
        else:
            cell = args.cell
            print(f"   Cell/Location name: {cell} (from CLI)")

        # Use environment from CLI (default: basement_parking)
        environment = args.environment
        if args.interactive:
            print(f"\n[ENV] Environment: {environment}")

        print("\n[MOVE] Beacon Status:")
        # Get beacon moving status (if not provided via CLI)
        if not args.beacon_moving:
            beacon_moving = get_yes_no_input("Is beacon moving?", default=False)
        else:
            beacon_moving = args.beacon_moving
            print(f"   Is beacon moving: Yes (from CLI)")

        print("\n[NOTE] Additional Notes:")
        # Get notes (if not provided via CLI)
        if not args.notes:
            notes = get_string_input("Notes (optional)", allow_empty=True)
        else:
            notes = args.notes
            print(f"   Notes: {notes} (from CLI)")

        # Show configuration summary
        print_configuration_summary(
            ground_truth_x=ground_truth_x,
            ground_truth_y=ground_truth_y,
            cell=cell if cell else None,
            beacon_moving=beacon_moving,
            environment=environment,
            notes=notes,
            samples=args.samples,
            interval=args.interval
        )

        # Ask for confirmation
        print()
        confirm = get_yes_no_input("Start collection?", default=True)

        if not confirm:
            print("\n[WARN] Collection cancelled by user")
            return 0

        print()

    else:
        # CLI-only mode (use args directly)
        ground_truth_x = args.ground_truth_x
        ground_truth_y = args.ground_truth_y
        cell = args.cell
        beacon_moving = args.beacon_moving
        environment = args.environment
        notes = args.notes

    # Create collector with parameters (from interactive or CLI)
    collector = ParallelDataCollector(
        output_file=args.output,
        verbose=not args.quiet,
        ground_truth_x=ground_truth_x,
        ground_truth_y=ground_truth_y,
        ground_truth_cell=cell,
        beacon_moving=beacon_moving,
        environment=environment,
        notes=notes
    )

    try:
        # Run data collection
        await collector.collect_data(
            num_samples=args.samples,
            interval_sec=args.interval
        )

        # Cleanup
        collector.cleanup()

        # Exit code based on success rate
        success_rate = collector.samples_collected / args.samples if args.samples > 0 else 0
        sys.exit(0 if success_rate >= 0.9 else 1)

    except KeyboardInterrupt:
        print("\n\n[WARN] Collection interrupted by user")
        collector.cleanup()
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[FATAL] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        collector.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nCollection cancelled by user")
        sys.exit(130)
