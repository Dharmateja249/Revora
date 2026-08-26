import sys
from pathlib import Path

# Add backend directory to sys.path so tests can import `app`
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
