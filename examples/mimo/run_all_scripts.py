import os
import subprocess

# List all Python files in the current directory
script_files = []
for name in os.listdir('.'):
    if name.endswith('.py') and name != 'run_all_scripts.py':
        script_files.append(name)

# Run each script
for script in script_files:
    print(f"Running {script}...")
    subprocess.run(['python', script])
