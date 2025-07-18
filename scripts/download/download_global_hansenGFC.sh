#!/bin/bash

#### NOTE: ALL COORDINATES REFER TO THE TOP LEFT CORNER OF THE GRANULES ####
#### USAGE NOTE: you modify the dest_dir and the input ranges for latitude and longitude in the script below ####

# Base URL
base_url="https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"

# Destination directory where files should be saved
dest_dir="/<DATA_DIR>/hansen_GFC-2023-v1.11"

# Create the destination directory if it does not exist
mkdir -p "${dest_dir}"

# Define file types to download
file_types=("lossyear" "datamask" "treecover2000" "gain")

# Function to convert coordinate labels (e.g., 50W, 20E) to numeric values
convert_to_numeric() {
  coord=$1
  if [[ $coord == *"W" ]]; then
    echo "-$(echo $coord | sed 's/W//')"
  elif [[ $coord == *"E" ]]; then
    echo "$(echo $coord | sed 's/E//')"
  elif [[ $coord == *"S" ]]; then
    echo "-$(echo $coord | sed 's/S//')"
  elif [[ $coord == *"N" ]]; then
    echo "$(echo $coord | sed 's/N//')"
  fi
}

# Function to generate sequences of multiples of 10, including boundaries
generate_sequence() {
  start=$1
  end=$2
  if [ $start -le $end ]; then
    echo $start $(seq $(( (start + 9) / 10 * 10 )) 10 $(( end / 10 * 10 ))) $end | tr ' ' '\n' | sort -n | uniq
  else
    echo $start $(seq $(( (start + 9) / 10 * 10 )) -10 $(( end / 10 * 10 ))) $end | tr ' ' '\n' | sort -nr | uniq
  fi
}

# Input ranges for global download
lon_start="180W"
lon_end="180E"
lat_start="60S"
lat_end="60N"

# Convert inputs to numeric values
lon_start_num=$(convert_to_numeric $lon_start)
lon_end_num=$(convert_to_numeric $lon_end)
lat_start_num=$(convert_to_numeric $lat_start)
lat_end_num=$(convert_to_numeric $lat_end)

# Loop through the specified ranges for latitude and longitude
for lat in $(generate_sequence $lat_start_num $lat_end_num); do
  for lon in $(generate_sequence $lon_start_num $lon_end_num); do
    # Determine if the longitude is East or West
    if [ $lon -lt 0 ]; then
      lon_label=$(printf "%03dW" $(echo "${lon}" | sed 's/-//'))
    else
      lon_label=$(printf "%03dE" ${lon})
    fi
    
    # Determine if the latitude is North or South
    if [ $lat -lt 0 ]; then
      lat_label=$(printf "%02dS" $(echo "${lat}" | sed 's/-//'))
    else
      lat_label=$(printf "%02dN" ${lat})
    fi

    # Loop through each file type and download
    for file_type in "${file_types[@]}"; do
      # Construct the filename and URL
      filename="Hansen_GFC-2023-v1.11_${file_type}_${lat_label}_${lon_label}.tif"
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
done