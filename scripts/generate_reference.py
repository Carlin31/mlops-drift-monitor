import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.config import settings
from app.core.model import generate_artifacts


def main() -> None:
    generate_artifacts(Path(settings.model_path), Path(settings.reference_data_path))
    print(f"Generated {settings.model_path} and {settings.reference_data_path}")


if __name__ == "__main__":
    main()
