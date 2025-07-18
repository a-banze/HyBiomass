#!/bin/bash

# Base URL
base_url="https://rcdata.nau.edu/geode_data/GEDIv002_L0204A_20190417to20230316_proc202312/tables"

# Destination directory where files should be saved
dest_dir="/<DATA_DIR>/GEDIlv002_L2L4a_global"

# Create the destination directory if it does not exist
mkdir -p "${dest_dir}"

# Loop through the ranges for North / West quadrant
for west in $(seq -w 0 180); do
  # Loop through the ranges for N
  for north in $(seq -w 0 52); do
    # Construct the filename and URL
    filename="gediv002_l2l4a_va_g${west}w${north}n.zip"
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

# Loop through the ranges for South / West quadrant
for west in $(seq -w 0 180); do
  # Loop through the ranges for S
  for south in $(seq -w 0 52); do
    # Construct the filename and URL
    filename="gediv002_l2l4a_va_g${west}w${south}s.zip"
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

# Loop through the ranges for North / East quadrant
for east in $(seq -w 0 180); do
  # Loop through the ranges for N
  for north in $(seq -w 0 52); do
    # Construct the filename and URL
    filename="gediv002_l2l4a_va_g${east}e${north}n.zip"
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

# Loop through the ranges for South / East quadrant
for east in $(seq -w 0 180); do
  # Loop through the ranges for S
  for south in $(seq -w 0 52); do
    # Construct the filename and URL
    filename="gediv002_l2l4a_va_g${east}e${south}s.zip"
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

# Count the number of files in the destination directory
num_files=$(find "$dest_dir" -type f | wc -l)
echo "The number of files in $dest_dir is $num_files."

# Check if the number of files is equal to 12,006
if [ "$num_files" -eq 12006 ]; then
  echo "This means all files were successfully downloaded."
  exit 0
else
  echo "The number of downloaded files should be 12,006. Exiting the script."
  exit 1
fi
