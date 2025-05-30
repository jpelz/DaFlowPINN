#!/bin/bash

#get simulation name

conf_file="/scratch/jpelz/ma-pinns/DA_CASE01.yaml"
simulation_name=$(grep '^name:' $conf_file | awk '{print $2}')
# Create directory if it does not exist
output_dir="/scratch/jpelz/ma-pinns/final_sims/$simulation_name"
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

new_file_name="$output_dir/$simulation_name.yaml"

echo "Simulation name: $simulation_name"
cp $conf_file $new_file_name

sbatch DA_CASE01.sbatch $new_file_name
