import os
import subprocess
from pathlib import Path
from subprocess import run

cfg_dir = Path("cfgs/batch1")
assert cfg_dir.is_dir()

error_count = 0

for cfg_file in os.listdir(cfg_dir):
    if cfg_file.endswith(".json"):
        cfg_path = cfg_dir / cfg_file
        print(f"Running {cfg_path}\n")
        cmd = f"python train.py --cfg {cfg_path}"
        try:
            run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            error_count += 1


print(f"\n\nAll executed with {error_count} errors\n\n")
