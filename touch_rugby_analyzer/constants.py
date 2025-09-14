from pathlib import Path
import rich

ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = ROOT / "assets"
rich.print(ROOT)