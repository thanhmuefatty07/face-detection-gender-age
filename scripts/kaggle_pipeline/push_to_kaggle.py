"""
Push to Kaggle CLI Runner
=========================
Pushes the notebook and metadata to Kaggle, monitors execution status,
and reports completion.
Usage:
    python scripts/kaggle_pipeline/push_to_kaggle.py [--push-only]
"""

import sys
import time
import subprocess
from pathlib import Path


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def main():
    notebook_dir = Path("notebooks").resolve()
    print(f"🚀 Pushing kernel from {notebook_dir} to Kaggle...")

    code, out, err = run_cmd(f"kaggle kernels push -p \"{notebook_dir}\"")
    if code != 0:
        print(f"❌ Error pushing kernel: {err or out}")
        sys.exit(1)

    print(f"✅ {out}")

    import json
    meta_path = notebook_dir / "kernel-metadata.json"
    kernel_id = "thanhmuefatty/face-dataset-mining-extreme-filtering-pipeline"
    if meta_path.exists():
        try:
            kernel_id = json.loads(meta_path.read_text(encoding="utf-8")).get("id", kernel_id)
        except Exception:
            pass
    print(f"⏳ Monitoring status for {kernel_id}...")

    if "--push-only" in sys.argv:
        print(f"Done (push-only mode). Check status anytime using: kaggle kernels status {kernel_id}")
        return

    for _ in range(60):
        code, out, err = run_cmd(f"kaggle kernels status {kernel_id}")
        if code == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {out}")
            if "complete" in out.lower():
                print("🎉 Kaggle Kernel execution completed successfully!")
                break
            elif "error" in out.lower():
                print("❌ Kernel failed with an error on Kaggle.")
                break
        time.sleep(20)


if __name__ == "__main__":
    main()
