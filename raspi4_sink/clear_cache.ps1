# Clear Python cache before data collection (Windows version)
# Run this BEFORE starting rpi_sink_parallel.py

Write-Host "Cleaning Python cache..." -ForegroundColor Cyan

# Remove __pycache__ directories
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | 
    ForEach-Object { 
        Remove-Item -Path $_.FullName -Recurse -Force
        Write-Host "  [OK] Removed: $($_.FullName)" -ForegroundColor Green
    }

# Remove .pyc files
Get-ChildItem -Path . -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue | 
    ForEach-Object { 
        Remove-Item -Path $_.FullName -Force
    }
Write-Host "  [OK] .pyc files removed" -ForegroundColor Green

# Remove .pyo files
Get-ChildItem -Path . -Recurse -File -Filter "*.pyo" -ErrorAction SilentlyContinue | 
    ForEach-Object { 
        Remove-Item -Path $_.FullName -Force
    }
Write-Host "  [OK] .pyo files removed" -ForegroundColor Green

Write-Host ""
Write-Host "Cache cleaned successfully!" -ForegroundColor Green
Write-Host "You can now run: python rpi_sink_parallel.py --samples 100 --interactive" -ForegroundColor Yellow
