#!/bin/bash

# Define the parameter arrays
VOL_FRACS=(0.2 0.25 0.3 0.35 0.4 0.45 0.5)
WEIGHTS=(0.05 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.95)

# Initialize a global counter for unique IDs
# (useful if you want a short integer ID in the filename)
COUNTER=0

# Loop 1: Iterate over Volume Fractions
for vf in "${VOL_FRACS[@]}"
do
    # Loop 2: Iterate over Weights
    for w in "${WEIGHTS[@]}"
    do
        # Increment global counter
        ((COUNTER++))

        # Create a unique filename encoding the parameters
        # Format: design_vf<volfrac>_w<weight>_id<counter>
        FILENAME="design_vf${vf}_w${w}_id${COUNTER}"

        # Submit the job with the 3 arguments: VOLFRAC, WEIGHT, FILENAME
        sbatch ./slurm_3d.sh "$vf" "$w" "$FILENAME"

        echo "Submitted job $COUNTER: VF=$vf, W=$w, File=$FILENAME"
    done
done