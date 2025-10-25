import os

# --------------------------------------------------------------
# Zaratan CPU Affinity (HPC)
# --------------------------------------------------------------
using_hpc = True
AVAILABLE_CPUS = 30
if using_hpc is True:
    slurm_cpu_ids = os.getenv('SLURM_JOB_CPUS_PER_NODE')
    if slurm_cpu_ids is not None:
        num_cpus = int(slurm_cpu_ids.split('(')[0])  # handles formats like "32(x2)"
        start_cpu_id = list(os.sched_getaffinity(0))[0]  # Get the first CPU ID
        allocated_cpus = list(range(start_cpu_id, start_cpu_id + num_cpus))
        print(f"Set CPU affinity to CPUs: {allocated_cpus}")
        os.sched_setaffinity(0, allocated_cpus)
        AVAILABLE_CPUS = num_cpus - 2

