#!/bin/bash

# Base URL
base_url="https://rcdata.nau.edu/geode_data/GEDIv002_L0204A_20190417to20230316_proc202312/tables"

# Destination directory where files should be saved
dest_dir="/<DATA_DIR>/GEDIlv002_L2L4a"

# Create the destination directory if it does not exist
mkdir -p "${dest_dir}"

# Loop through the ranges for W
for west in $(seq -w 1 11); do
  # Loop through the ranges for N
  for north in $(seq 35 52); do
    # Construct the filename and URL
    filename="gediv002_l2l4a_va_g$(printf "%03d" ${west})w${north}n.zip"
    url="${base_url}/${filename}"
    full_path="${dest_dir}/${filename}"

    # Check if the file already exists
    if [ -f "${full_path}" ]; then
      echo "File ${filename} already exists in ${dest_dir}, skipping download."
    else
      # Download the file and save it with the original filename
      echo "Downloading ${filename} to ${dest_dir}..."
      curl -f -o "${full_path}" "${url}" || echo "File ${filename} not found, skipping..."
    fi
  done
done

# Loop through the ranges for E
for east in $(seq -w 0 32); do
  # Loop through the ranges for N
  for north in $(seq 35 52); do
    # Construct the filename and URL
    filename="gediv002_l2l4a_va_g$(printf "%03d" ${east})e${north}n.zip"
    url="${base_url}/${filename}"
    full_path="${dest_dir}/${filename}"

    # Check if the file already exists
    if [ -f "${full_path}" ]; then
      echo "File ${filename} already exists in ${dest_dir}, skipping download."
    else
      # Download the file and save it with the original filename
      echo "Downloading ${filename} to ${dest_dir}..."
      curl -f -o "${full_path}" "${url}" || echo "File ${filename} not found, skipping..."
    fi
  done
done
