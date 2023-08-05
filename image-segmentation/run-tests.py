import sys
import subprocess
from datetime import datetime

original_stdout = sys.stdout

arr = ["pytorch-test.py", "tensorflow-test.py", "keras-test.py", "numpy-test.py"]

with open('time.txt', 'a') as f:
    sys.stdout = f
    print(f"\nAutomated test: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

sys.stdout = original_stdout

for f in arr:
    print(f"Running {f}...")
    subprocess.run(["python", f])
    print(f"{f} executed successfully.\n")
    
  