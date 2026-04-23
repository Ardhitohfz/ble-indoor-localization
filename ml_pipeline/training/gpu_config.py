import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Any
import numpy as np
import lightgbm as lgb

from config_ml import GPU_CONFIG
from core.logger import get_logger

logger = get_logger(__name__)

def detect_gpu() -> Dict[str, Any]:
    if not GPU_CONFIG["use_gpu"]:
        logger.info("GPU disabled in config, using CPU")
        return {"device": "cpu", "num_threads": -1}

    try:
        logger.info("Testing CUDA GPU support...")
        test_data = lgb.Dataset(
            np.random.rand(100, 10), label=np.random.randint(0, 2, 100)
        )
        test_params = {
            "device": "gpu",
            "gpu_platform_id": GPU_CONFIG["gpu_platform_id"],
            "gpu_device_id": GPU_CONFIG["gpu_device_id"],
            "objective": "binary",
            "verbose": -1,
        }

        model = lgb.train(test_params, test_data, num_boost_round=1)

        logger.info("[OK] CUDA GPU detected and working")
        return {
            "device": "gpu",
            "gpu_platform_id": GPU_CONFIG["gpu_platform_id"],
            "gpu_device_id": GPU_CONFIG["gpu_device_id"],
            "gpu_use_dp": GPU_CONFIG["gpu_use_dp"],
        }

    except Exception as e_cuda:
        logger.info(f"CUDA not available: {str(e_cuda)[:100]}")

    try:
        logger.info("Testing OpenCL GPU support...")
        test_params = {
            "device": "gpu",
            "gpu_platform_id": GPU_CONFIG["gpu_platform_id"],
            "gpu_device_id": GPU_CONFIG["gpu_device_id"],
            "gpu_use_dp": False,
            "objective": "binary",
            "verbose": -1,
        }

        model = lgb.train(test_params, test_data, num_boost_round=1)

        logger.info("[OK] OpenCL GPU detected and working")
        return {
            "device": "gpu",
            "gpu_platform_id": GPU_CONFIG["gpu_platform_id"],
            "gpu_device_id": GPU_CONFIG["gpu_device_id"],
            "gpu_use_dp": False,
        }

    except Exception as e_opencl:
        logger.info(f"OpenCL not available: {str(e_opencl)[:100]}")

    logger.warning("[WARNING] GPU not available, using CPU (slower but works)")
    return {"device": "cpu", "num_threads": -1}

def get_gpu_params(base_params: Dict[str, Any]) -> Dict[str, Any]:
    gpu_config = detect_gpu()

    params = base_params.copy()

    if gpu_config["device"] == "gpu":
        params.update(
            {
                "device": "gpu",
                "gpu_platform_id": gpu_config.get("gpu_platform_id", 0),
                "gpu_device_id": gpu_config.get("gpu_device_id", 0),
                "gpu_use_dp": gpu_config.get("gpu_use_dp", False),
                "max_bin": GPU_CONFIG["max_bin"],
                "num_threads": GPU_CONFIG.get("num_threads", 1),
            }
        )

        params.pop("n_jobs", None)
        params.pop("force_col_wise", None)
        params.pop("force_row_wise", None)

        logger.info(
            f"[OK] GPU training enabled (device={gpu_config['gpu_device_id']}, "
            f"max_bin={GPU_CONFIG['max_bin']}, num_threads={GPU_CONFIG.get('num_threads', 1)})"
        )

    else:
        params.update(
            {
                "device": "cpu",
                "num_threads": -1,
            }
        )

        if "n_jobs" not in params:
            params["n_jobs"] = -1

        logger.info("Using CPU training (all cores)")

    return params

def is_gpu_available() -> bool:
    config = detect_gpu()
    return config["device"] == "gpu"

def get_device_info() -> str:
    config = detect_gpu()

    if config["device"] == "gpu":
        platform = "CUDA" if config.get("gpu_use_dp", False) else "OpenCL"
        device_id = config.get("gpu_device_id", 0)
        return f"GPU ({platform}, device {device_id}) - 4-5× faster training"
    else:
        return "CPU (all cores) - GPU not available"

if __name__ == "__main__":
    print("=" * 60)
    print("GPU CONFIGURATION TEST")
    print("=" * 60)

    print("\n1. Testing GPU detection:")
    config = detect_gpu()
    print(f"   Device: {config['device']}")
    if config["device"] == "gpu":
        print(f"   Platform ID: {config.get('gpu_platform_id', 'N/A')}")
        print(f"   Device ID: {config.get('gpu_device_id', 'N/A')}")
        print(f"   Double Precision: {config.get('gpu_use_dp', 'N/A')}")

    print("\n2. Testing parameter generation:")
    base_params = {
        "objective": "multiclass",
        "num_class": 25,
        "learning_rate": 0.0005,
        "num_leaves": 31,
    }

    optimized_params = get_gpu_params(base_params)
    print(f"   Device: {optimized_params.get('device', 'N/A')}")
    print(f"   Max Bin: {optimized_params.get('max_bin', 'N/A')}")

    print("\n3. Device information:")
    print(f"   {get_device_info()}")

    print("\n4. Quick availability check:")
    if is_gpu_available():
        print("   [OK] GPU is available - expect 4-5× faster training!")
    else:
        print("   [WARNING] GPU not available - using CPU (slower but works)")

    print("\n" + "=" * 60)