import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

CACHE_LOCATION =  ROOT / ".miniab"
if not os.path.exists(CACHE_LOCATION):
    os.mkdir(CACHE_LOCATION)
