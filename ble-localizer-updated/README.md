# Firmware ESP32 (BLE Localizer)

Folder ini berisi firmware PlatformIO untuk:

- **Beacon** (`env:beacon`): memancarkan BLE advertisement.
- **Anchor** (`env:anchor_a` sampai `env:anchor_d`): membaca RSSI beacon dan expose data via GATT.

## Struktur Singkat

```text
ble-localizer-updated/
├── platformio.ini
├── firmware/
│   ├── beacon/
│   └── anchor/
└── shared/
```

## Build

```bash
cd ble-localizer-updated
pio run --environment beacon
pio run --environment anchor_a
pio run --environment anchor_b
pio run --environment anchor_c
pio run --environment anchor_d
```

## Upload

`platformio.ini` sudah berisi `upload_port` per environment. Pastikan port sesuai perangkat sebelum upload.

```bash
pio run --environment beacon --target upload
pio run --environment anchor_a --target upload
```
