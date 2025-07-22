import os
import subprocess

# List all Python files in the current directory
script_files = [f for f in os.listdir('.') if f.endswith('.py') and f != 'run_all_scripts.py']

# Run each script
for script in script_files:
    print(f"Running {script}...")
    subprocess.run(['python', script])
