#!/bin/bash

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
  echo "Cloning the repository from $repo_url with branch $repo_branch"
  if git clone -b "$repo_branch" "$repo_url" "$dir_name"; then
    echo "Repository cloned successfully into $dir_name!"
  else
    echo "Failed to clone the repository."
    exit 1
  fi
}

fix_cuda_issues() {
  log() {
    echo "$(date +"%Y-%m-%d %T") : $1"
  }

  error_exit() {
    echo "$(date +"%Y-%m-%d %T") : ERROR: $1"
    exit 1
  }

  # Function to set environment variables for CUDA
  set_cuda_env() {
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    export CUDA_ROOT_DIR=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    log "CUDA environment variables set."
  }

  # Function to remove existing NVIDIA drivers and CUDA installations
  remove_nvidia_drivers() {
    log "Removing existing NVIDIA drivers and CUDA installations..."
    sudo apt remove --purge -y nvidia* --allow-change-held-packages || error_exit "Failed to remove NVIDIA packages"
    sudo apt autoremove -y || error_exit "Failed to autoremove packages"
  }

  # Function to check if NVIDIA files are totally removed
  check_nvidia_removal() {
    log "Checking if NVIDIA files are completely removed..."
    if dpkg -l | grep -i nvidia; then
      error_exit "NVIDIA files still found. Please remove manually and retry."
    else
      log "NVIDIA files successfully removed."
    fi
  }

  # Function to install CUDA
  install_cuda() {
    log "Installing CUDA..."
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run || error_exit "Failed to download CUDA installer"
    sudo sh cuda_12.4.0_550.54.14_linux.run --silent --driver --toolkit || error_exit "Failed to install CUDA"
  }

  # Function to install PyCUDA
  install_pycuda() {
    log "Installing PyCUDA..."
    sudo apt-get install -y python3-pycuda || error_exit "Failed to install PyCUDA"
  }

  # Function to install NVIDIA Container Toolkit
  install_nvidia_toolkit() {
    log "Installing NVIDIA Container Toolkit..."
    sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg || error_exit "Failed to add NVIDIA GPG key"
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null || error_exit "Failed to add NVIDIA Container Toolkit repository"
    sudo apt-get update || error_exit "Failed to update package lists"
    sudo apt-get install -y nvidia-container-toolkit || error_exit "Failed to install NVIDIA Container Toolkit"
  }

  # Check if CUDA is installed
  if command -v nvcc &>/dev/null; then
    log "nvcc found. Setting CUDA environment variables..."
    set_cuda_env
  else
    log "nvcc not found. Assuming CUDA is not installed properly."

    # Remove existing NVIDIA drivers and install CUDA
    remove_nvidia_drivers
    check_nvidia_removal
    install_cuda

    # Set environment variables after CUDA installation
    set_cuda_env
  fi

  # Check if PyCUDA is installed
  if python3 -c "import pycuda" &>/dev/null; then
    log "PyCUDA is already installed."
  else
    log "PyCUDA not found. Installing PyCUDA..."
    install_pycuda
  fi

  # Install NVIDIA Container Toolkit
  install_nvidia_toolkit
  uninstall_numpy

  log "Script execution completed."
}

main() {
  local dir_name="inference_results_v3.0"
  REPO_URL="https://github.com/brahmGAN/inference_results_v3.0.git"
  local mode="${1:-full}"

  fix_cuda_issues

  nvidia_output=$(nvidia-smi)
  REPO_BRANCH=$(echo "$nvidia_output" | grep -oP '(?<=\|   ).*(?=  [0-9])' | head -n 1 | awk '{print $3}' | sed 's/ /_/')-full
  echo "$REPO_BRANCH"

  if [ "$mode" = "full" ]; then
    #    pip3 install numpy 1.23.1
    sudo apt install python3-numpy
    clone_repository "$dir_name" "$REPO_URL" "$REPO_BRANCH"
  fi

  cd "$dir_name"
  echo $(pwd)

  if [ "$mode" = "full" ]; then
    bash script.sh
  else
    bash script_quick.sh
  fi
}

# Call the main function to execute the script
main "$@"
