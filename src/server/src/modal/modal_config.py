# agents/modal_config.py
"""
Configuration settings for Modal experiment execution
"""

# Modal app configuration
MODAL_APP_NAME = "ml-experiment-runner"

# Resource configuration
GPU_CONFIG = "any"  # Options: "any", "T4", "A10G", "A100", "H100", or None for CPU
MEMORY_GB = 8
TIMEOUT_SECONDS = 1800  # 30 minutes

# Volume configuration
VOLUME_NAME = "experiment-artifacts"

# Python packages to install in the Modal image
REQUIRED_PACKAGES = [
    "torch",
    "torchvision", 
    "scikit-learn",
    "matplotlib",
    "numpy",
    "pandas",
    "tqdm",
    "seaborn",
    "plotly",
    "scipy"
]

# Default experiment configuration
DEFAULT_EXPERIMENT_CONFIG = {
    "platform": "modal",
    "execution_mode": "cloud",
    "resource_type": "gpu" if GPU_CONFIG else "cpu",
    "memory_gb": MEMORY_GB,
    "timeout_seconds": TIMEOUT_SECONDS
} 