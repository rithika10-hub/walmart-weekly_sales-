import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log filename with timestamp
LOG_FILE = f"{LOG_DIR}/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure the logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Also log to console (optional but useful)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Add the console logger to the root logger
logging.getLogger().addHandler(console)
