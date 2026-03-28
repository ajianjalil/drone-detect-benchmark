#!/bin/bash
#SBATCH --job-name=drone_detect      # Job name
#SBATCH --output=slurm-%j.out        # Standard output and error log (%j inserts job ID)
#SBATCH --nodes=1                    # Run on 1 node
#SBATCH --ntasks-per-node=1          # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=8GB                    # Job memory request
#SBATCH --time=00:30:00              # Time limit hrs:min:sec

# --- Environment Setup and Execution ---

echo "Starting job on $(hostname)"
echo "Current directory: $(pwd)"

# 1. Load the specific Python module (if necessary on your cluster)
# Your cluster might require loading a specific python environment module first.
# Uncomment the line below if your cluster uses 'module load' for Python:
# module load python/3.12 

# 2. Create the Venv (only needs to run once per cluster/directory)
# You might want to run this interactively ONCE, or comment it out after the first run.
python3 -m venv venv2

# 3. Activate the virtual environment
source venv2/bin/activate

pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121



# 4. Install dependencies (only needs to run once per cluster/directory)
# Uncomment the line below if dependencies haven't been installed yet:
pip install -r requirements.txt --no-deps


# 5. Run your main Python application
# Replace 'main.py' with the actual entry point of your application
echo "Running Python application..."
python main.py

echo "Job finished."

