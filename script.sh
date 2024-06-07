#!/bin/bash

run_custom_system_script() {
  # Run the Python script and capture the output
  local script_output
  nvidia_output=$(nvidia-smi)
  gpu_name=$(echo "$nvidia_output" | grep -oP '(?<=\|   ).*(?=  [0-9])' | head -n 1 | awk '{print $2, $3}' | sed 's/ /_/')
  echo "$gpu_name"
  script_output=$(echo "$gpu_name" | python3 -m scripts.custom_systems.add_custom_system)
  system_id=$(echo "$script_output" | grep -oP 'NVIDIA\s[A-Z0-9]+' | sed 's/ /_/')
  echo $system_id
  # Generate content for the .py file
  local py_content
  py_content="import os\n"
  py_content+="import sys\n"
  py_content+="sys.path.insert(0, os.getcwd())\n\n"
  py_content+="from code.common.constants import Benchmark, Scenario\n"
  py_content+="from code.common.systems.system_list import KnownSystem\n"
  py_content+="from configs.configuration import *\n"
  py_content+="from configs.bert import GPUBaseConfig, CPUBaseConfig\n\n"
  py_content+="class OfflineGPUBaseConfig(GPUBaseConfig):\n"
  py_content+="    scenario = Scenario.Offline\n"
  py_content+="    gpu_copy_streams = 2\n"
  py_content+="    gpu_inference_streams = 2\n"
  py_content+="    enable_interleaved = False\n"
  py_content+="    use_small_tile_gemm_plugin = True\n"
  py_content+="    gpu_batch_size = 1024\n"
  py_content+="    offline_expected_qps = 3400\n"
  py_content+="    workspace_size = 7516192768\n\n"
  py_content+="@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)\n"
  py_content+="class ${system_id}(OfflineGPUBaseConfig):\n"
  py_content+="    system = KnownSystem.${system_id}\n"

  # Write content to the .py file
  local py_file_path="configs/bert/Offline/__init__.py"
  #local py_file_path="configs/bert/Offline/__init__.py"
  echo -e "$py_content" >|"$py_file_path"

  echo "Updated .py file at $py_file_path"

  # Path to your configuration file
  CONFIG_FILE_PATH="configs/bert/Offline/custom.py"

  # Clear the contents of the configuration file
  echo "" >|$CONFIG_FILE_PATH
}

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
  #  local full_path="$dir_name/closed/$hardware_name"
  local full_path="closed/$hardware_name"

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
  local dir_name="inference_results_v3.0"
  export MLPERF_SCRATCH_PATH=$(pwd)/scratch

  ensure_python_dependencies
  read_yaml

  #  clone_repository "$dir_name" "$REPO_URL" "$REPO_BRANCH"
  #  create_directory_structure
  change_to_hardware_dir "$dir_name" "$HARDWARE"
  set_timezone "$TIMEZONE"

  # Convert benchmarks into an array
  IFS=' ' read -r -a BENCHMARK_MODELS <<<"$BENCHMARKS"

  # Run the nvidia-smi mig -lgip command and store the output in a variable
  output=$(nvidia-smi mig -lgip)

  # Use awk to extract the last MIG ID from the output

  # Extracting the last GPU name
  last_gpu_name=$(echo "$output" | grep -oP 'MIG \d+g\.\d+gb(?=\s+\d+)' | tail -1)

  last_gpu_name=$(echo "$last_gpu_name" | grep -oE "[^ ]+$")
  echo "$last_gpu_name"

  # Check if last_mig_id is empty, indicating no MIG device found
  if [ -z "$last_gpu_name" ]; then
    echo "Error: No MIG device found."
    exit 1
  fi

  # Run the nvidia-smi mig -cgi command with the last MIG ID
  result=$(sudo nvidia-smi mig -cgi "$last_gpu_name" -C 2>&1)

  # Check if the command was successful
  if [[ $result == *"Successfully"* ]]; then
    echo "Successfully created GPU instance ID $last_gpu_name"
    # Continue with your script here
  else
    echo "Error: Failed to create GPU instance ID $last_gpu_name"
    exit 1
  fi

  systemctl restart docker
  rm -rf code/common/systems/custom_list.py
  run_custom_system_script

  # Loop over the benchmarks
  for model in "${BENCHMARK_MODELS[@]}"; do
    make prebuild DOCKER_COMMAND="make download_data BENCHMARKS='$model'"
    make prebuild DOCKER_COMMAND="make download_model BENCHMARKS='$model'"
    make prebuild DOCKER_COMMAND="make preprocess_data BENCHMARKS='$model'"
    make prebuild DOCKER_COMMAND="make clean"
    make prebuild DOCKER_COMMAND="make build"
    output=$(make prebuild DOCKER_COMMAND="make run RUN_ARGS='--benchmarks=$model --scenarios=offline'")
    # Check if "Result is : VALID" is in the output
    if echo "$output" | grep -q "Result is : VALID"; then
      echo "True"
    else
      echo "False"
    fi

    sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
  done

}

# Call the main function to execute the script
main
