# run etl scripts
import subprocess
scripts = [
    "/app/scripts/04_train_model.py"
]

def run_script(script_path):
    print(f"\n🚀 Running: {script_path}")
    result = subprocess.run(f"python {script_path}", shell=True)
    if result.returncode == 0:
        print(f"✅ Completed: {script_path}")
    else:
        print(f"❌ Failed: {script_path}")

if __name__ == "__main__":
    for script in scripts:
        run_script(script)