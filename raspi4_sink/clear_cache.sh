#!/bin/bash
# Clear Python cache before data collection
# Run this BEFORE starting rpi_sink_parallel.py

echo "Cleaning Python cache..."

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "  [OK] __pycache__ directories removed"

# Remove .pyc files
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "  [OK] .pyc files removed"

# Remove .pyo files (optimized bytecode)
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "  [OK] .pyo files removed"

echo ""
echo "Cache cleaned successfully!"
echo "You can now run: python3 rpi_sink_parallel.py --samples 100 --interactive"
