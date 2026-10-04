from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eager.verify_smollm_fx import main


if __name__ == "__main__":
    main()
