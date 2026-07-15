"""Put the `src/` layout on sys.path so `tools/*.py` can import `mvface`without requiring an editable install. Import this first in any entry script.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
