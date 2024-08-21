#!/bin/bash

# Function to ensure Python and PyYAML are installed
ensure_python_dependencies() {
  # Check for Python3
  if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Please install Python3."
    exit 1
  fi

  # Check for PyYAML and install if not present
  if ! python3 -c "import yaml" 2>/dev/null; then
    echo "PyYAML is not installed. Installing..."
    pip3 install pyyaml || {
      echo "Failed to install PyYAML."
      exit 1
    }
  fi

}

# Function to read values including a simple list of benchmarks from a YAML file using Python
read_yaml() {
  # Use Python to parse YAML and extract values including benchmarks as a space-separated string
  IFS=' ' read -r REPO_URL REPO_BRANCH HARDWARE TIMEZONE BENCHMARKS <<<$(python3 -c "
import yaml, sys
data = yaml.safe_load(sys.stdin)

# Extract repository URL, hardware, and timezone
repo_url = data.get('repository', {}).get('url', '')
repo_branch = data.get('repository', {}).get('branch', '')
hardware = data.get('hardware', '')
timezone = data.get('timezone', '')

# Extract benchmarks as a list of names, expecting simple strings in the list
benchmarks = ' '.join(data.get('benchmark', []))

# Print extracted values in a single line for Bash to read
print(repo_url, repo_branch, hardware, timezone, benchmarks)
" <values.yaml)

  # Check if the repository URL was successfully extracted
  if [ -z "$REPO_URL" ]; then
    echo "Failed to extract repository URL. Check your values.yaml file."
    exit 1
  fi

  # Check if the repository Branch was successfully extracted
  if [ -z "$REPO_BRANCH" ]; then
    echo "Failed to extract repository Branch. Check your values.yaml file."
    exit 1
  fi

  # Check if the Hardware name was successfully extracted
  if [ -z "$HARDWARE" ]; then
    echo "Failed to extract Hardware Name. Check your values.yaml file."
    exit 1
  fi

  # Check if the Timezone name was successfully extracted
  if [ -z "$TIMEZONE" ]; then
    echo "Failed to extract Timezone Name. Check your values.yaml file."
    exit 1
  fi

  # Check if Benchmark models were successfully extracted
  if [ -z "$BENCHMARKS" ]; then
    echo "Failed to extract Benchmark Models. Check your values.yaml file."
    exit 1
  else
    # Displaying benchmarks for debugging
    echo "Benchmark Models extracted: $BENCHMARKS"
  fi
}

set_timezone() {
  local timezone=$1

  # Ensure the timezone parameter is not empty
  if [[ -z "$timezone" ]]; then
    echo "Usage: set_timezone <Timezone>"
    return 1
  fi

  # Check if /etc/timezone is currently a directory and remove it if it is
  if [[ -d "/etc/timezone" ]]; then
    sudo rm -r /etc/timezone
  fi

  # Create /etc/timezone file and set the desired timezone
  echo "$timezone" | sudo tee /etc/timezone >/dev/null

  # Update /etc/localtime to point to the correct timezone data file
  sudo ln -sf "/usr/share/zoneinfo/$timezone" /etc/localtime

  # Output the current settings to confirm changes
  echo "Timezone set to $(cat /etc/timezone)"
  echo "/etc/localtime is linked to $(readlink /etc/localtime)"
}

# Function to clone the repository into a specified directory
clone_repository() {
  local dir_name=$1
  local repo_url=$2
  local repo_branch=$3

  # Check if the directory exists
  if [ -d "$dir_name" ]; then
    echo "Directory $dir_name exists. Removing..."
    rm -rf "$dir_name"
  fi

  # Clone the repository
  echo "Cloning the repository from $repo_url..."
  if git clone -b "$repo_branch" "$repo_url" "$dir_name"; then
    echo "Repository cloned successfully into $dir_name!"
  else
    echo "Failed to clone the repository."
    exit 1
  fi
}

# Function to create directory structure
create_directory_structure() {
  # Use an environment variable for the base directory
  local base_dir="${MLPERF_SCRATCH_PATH}"

  # Define the subdirectories
  local sub_dirs=("data" "models" "preprocessed_data")

  # Check if the base directory is set
  if [ -z "$base_dir" ]; then
    echo "MLPERF_SCRATCH_PATH is not set. Please set the environment variable."
    exit 1
  fi

  # Check if the base directory exists, if not, create it
  [ -d "$base_dir" ] || mkdir -p "$base_dir"

  # Create subdirectories in one line using mkdir's ability to create multiple directories
  mkdir -p "$base_dir/data" "$base_dir/models" "$base_dir/preprocessed_data"

  # Confirmation message
  echo "Directories set up under $base_dir:"
  for dir in "${sub_dirs[@]}"; do
    echo "  $base_dir/$dir"
  done
}

# change dir to hardware folder
change_to_hardware_dir() {
  local dir_name=$1
  local hardware_name=$2

  # Construct the full path to the hardware directory
  local full_path="$dir_name/closed/$hardware_name"

  # Check if the directory exists
  if [ -d "$full_path" ]; then
    echo "Changing to directory: $full_path"
    cd "$full_path"
  else
    echo "Directory $full_path does not exist."
    exit 1
  fi
}

# Main function to orchestrate the steps
main() {
  export MLPERF_SCRATCH_PATH=$(pwd)/scratch

  ensure_python_dependencies
  read_yaml
  local dir_name="inference_results_v3.0"
  #  clone_repository "$dir_name" "$REPO_URL" "$REPO_BRANCH"
  create_directory_structure
  change_to_hardware_dir "$dir_name" "$HARDWARE"
  set_timezone "$TIMEZONE"

  # Convert benchmarks into an array
  IFS=' ' read -r -a BENCHMARK_MODELS <<<"$BENCHMARKS"

  # Loop over the benchmarks
  for model in "${BENCHMARK_MODELS[@]}"; do
    #    make prebuild DOCKER_COMMAND="make download_data BENCHMARKS='$model'"
    #    make prebuild DOCKER_COMMAND="make download_model BENCHMARKS='$model'"
    #    make prebuild DOCKER_COMMAND="make preprocess_data BENCHMARKS='$model'"
    #    make prebuild DOCKER_COMMAND="make clean"
    #    make prebuild DOCKER_COMMAND="make build"
    make prebuild DOCKER_COMMAND="make run RUN_ARGS='--benchmarks=$model'"

  done

}

# Call the main function to execute the script
main
