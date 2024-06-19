#!/usr/bin/python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import importlib.util
import re
import shutil
import textwrap
from copy import deepcopy
from importlib import import_module, reload

# We want system_list to be easily reloadable. Import the entire module, instead of directly importing member variables.
import code.common.systems.system_list
DETECTED_SYSTEM = code.common.systems.system_list.DETECTED_SYSTEM
MATCHED_SYSTEM = code.common.systems.system_list.MATCHED_SYSTEM
KnownGPU = code.common.systems.system_list.KnownGPU

from code.common.systems.base import obj_to_codestr
from code.common.fields import get_applicable_fields
from code.common.constants import *


__doc__ = """This script creates a custom system definition within the MLPerf Inference codebase that matches the
hardware specifications of the system that it is run on. The script then does the following:

    - Backs up NVIDIA's workload configuration files
    - Creates new workload configuration files (configs/<Benchmark name>/<Scenario>/__init__.py) with dummy values
        - The user should fill out these dummy values with the correct values
"""

SYSTEM_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
CUSTOM_LIST_MODULE_PATH = "code.common.systems.custom_list"
CUSTOM_LIST_FILE_PATH = "code/common/systems/custom_list.py"
CUSTOM_LIST_STUB_FILE = "scripts/custom_systems/stubs/custom_system.py.stub"
CONFIG_FILE_FORMAT = "configs/{benchmark}/{scenario}/__init__.py"
CUSTOM_CONFIG_FILE_FORMAT = "configs/{benchmark}/{scenario}/custom.py"
CUSTOM_CONFIG_FILE_HEADER = ["# Generated file by scripts/custom_systems/add_custom_system.py\n",
                             f"# Contains configs for all custom systems in {CUSTOM_LIST_FILE_PATH}\n",
                             "\n",
                             "from . import *\n",
                             "\n",
                             "\n"]

HAS_HIGHACC = [Benchmark.UNET3D, Benchmark.BERT, Benchmark.DLRM]
HAS_TRITON = [Benchmark.UNET3D, Benchmark.BERT, Benchmark.DLRM, Benchmark.ResNet50, Benchmark.Retinanet]


class KnownGPUSpoof:
    """Spoofs a SystemConfiguration and allows a codestring to be generated based on a KnownGPU enum member"""

    def __init__(self, original, equivalent_known_gpu):
        for field_name in original.__dataclass_fields__.keys():
            object.__setattr__(self, field_name, original.__getattribute__(field_name))

        self._known_gpu_enum_member = equivalent_known_gpu

    def codestr(self):
        return f"{self._known_gpu_enum_member}.value"

    def __hash__(self):
        return hash(self.codestr())


def reload_system_list():
    """Reload the code.common.systems.system_list import to re-import the KnownSystem Enum"""
    global DETECTED_SYSTEM, MATCHED_SYSTEM, KnownGPU

    # The filechange is actually in custom_list which may not have existed before. Force the import and reload it.
    if importlib.util.find_spec(CUSTOM_LIST_MODULE_PATH) is not None:
        custom_list_module = importlib.import_module(CUSTOM_LIST_MODULE_PATH)
        reload(custom_list_module)
    reload(code.common.systems.system_list)
    system_list = import_module("code.common.systems.system_list")
    DETECTED_SYSTEM = system_list.DETECTED_SYSTEM
    MATCHED_SYSTEM = system_list.MATCHED_SYSTEM
    KnownGPU = system_list.KnownGPU


def generate_config(benchmark, scenario, system):
    custom_config_file = CUSTOM_CONFIG_FILE_FORMAT.format(benchmark=benchmark.valstr(), scenario=scenario.valstr())
    lines = []
    if os.path.exists(custom_config_file):
        with open(custom_config_file) as custom_file:
            lines = custom_file.readlines()

    if len(lines) == 0:
        lines += CUSTOM_CONFIG_FILE_HEADER

    # Figure out which configs to add based on the benchmark
    workloads = [(G_DEFAULT_HARNESS_TYPES[benchmark], AccuracyTarget.k_99, PowerSetting.MaxP)]
    if benchmark in HAS_HIGHACC:
        for workload in workloads[:]:
            workloads.append((workload[0], AccuracyTarget.k_99_9, workload[2]))
    if benchmark in HAS_TRITON:
        for workload in workloads[:]:
            workloads.append((HarnessType.Triton, workload[1], workload[2]))

    system_id = system.system_id
    config_id = system_id.upper()

    def _get_config_class_name(workload):
        config_class_name = config_id
        if workload[1] == AccuracyTarget.k_99_9:
            config_class_name += "_HighAccuracy"
        if workload[0] == HarnessType.Triton:
            config_class_name += "_Triton"
        return config_class_name

    # Generate each config
    for i, workload in enumerate(workloads):
        # Generate the ConfigRegistry line
        lines.append(f"@ConfigRegistry.register({workload[0]}, {workload[1]}, {workload[2]})\n")

        # Generate the class name
        config_class_name = _get_config_class_name(workload)
        if i == 0:
            base_class_name = f"{scenario.valstr()}GPUBaseConfig"
        elif workload[0] == HarnessType.Triton and workload[1] == AccuracyTarget.k_99_9:
            # For HighAcc Triton, inherit from non-Triton HighAcc
            base_class_name = _get_config_class_name((workloads[0][0], AccuracyTarget.k_99_9, workload[2]))
        else:
            # For HighAcc and Triton, inherit from the original base system class
            base_class_name = _get_config_class_name(workloads[0])
        lines.append(f"class {config_class_name}({base_class_name}):\n")

        # For the base system config, add the 'system' key
        if i == 0:
            lines.append(f"    system = KnownSystem.{system_id}\n")

        if workload[1] == AccuracyTarget.k_99_9 and workload[0] != HarnessType.Triton:  # Handle HighAcc
            if benchmark == Benchmark.BERT:  # BERT HighAcc uses FP16
                lines.append("    precision = \"fp16\"\n")
                if scenario == Scenario.Offline:
                    lines.append(f"    offline_expected_qps = {base_class_name}.offline_expected_qps / 2\n")
                elif scenario == Scenario.Server:
                    lines.append(f"    server_target_qps = {base_class_name}.server_target_qps / 2\n")
                elif scenario == Scenario.SingleStream:
                    lines.append(f"    single_stream_expected_latency_ns = {base_class_name}.single_stream_expected_latency_ns * 2\n")
            else:
                lines.append("    pass\n")
        elif workload[0] == HarnessType.Triton:  # Handle Triton and HighAcc Triton
            lines.append("    use_triton = True\n")

        if workload[1] == AccuracyTarget.k_99:  # For the base system class and base Triton class, generate the fields
            gen_mandatory, gen_opt = get_applicable_fields(Action.GenerateEngines, benchmark, scenario, system, WorkloadSetting(*workload))
            run_mandatory, run_opt = get_applicable_fields(Action.RunHarness, benchmark, scenario, system, WorkloadSetting(*workload))

            mandatory = dict()
            for field in gen_mandatory + run_mandatory:
                if field.name not in mandatory:
                    mandatory[field.name] = field
            optional = dict()
            for field in gen_opt + run_opt:
                if field.name not in optional:
                    optional[field.name] = field

            # Remove some of the ones that shouldn't be in configs. Some of these are Loadgen fields that shouldn't be
            # modified. Others are fields set by the script or by the base config.
            skip = ["accuracy_log_rng_seed",
                    "benchmark",
                    "data_dir",
                    "devices",
                    "disable_log_copy_summary_to_stdout",
                    "fast",
                    "force_calibration",
                    "log_copy_detail_to_stdout",
                    "log_dir",
                    "log_enable_trace",
                    "log_mode",
                    "log_mode_async_poll_interval_ms",
                    "logfile_prefix_with_datetime",
                    "logfile_suffix",
                    "max_duration",
                    "max_query_count",
                    "min_duration",
                    "min_query_count",
                    "mlperf_conf_path",
                    "performance_sample_count",
                    "preprocessed_data_dir",
                    "qsl_rng_seed",
                    "sample_index_rng_seed",
                    "scenario",
                    "system",
                    "test_mode",
                    "use_triton",
                    "user_conf_path",
                    "verbose",
                    "verbose_nvtx"]

            def add_field_lines(field_dict):
                for field_name in sorted(field_dict.keys()):
                    if field_name in skip:
                        continue
                    field = field_dict[field_name]
                    default_value = field.value_type()  # Use default constructor
                    lines.append(f"    {field_name}: {field.value_type.__name__} = {repr(default_value)}\n")

            lines.append("\n")
            lines.append("    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.\n")
            lines.append("    # Please see NVIDIA's submission config files for example values and which fields to keep.\n")
            lines.append("    # Required fields (Must be set or inherited to run):\n")
            add_field_lines(mandatory)
            lines.append("\n    # Optional fields:\n")
            add_field_lines(optional)

        lines.append("\n\n")

    if os.path.exists(custom_config_file):
        print("=> Backing up original file to {custom_config_file}.backup...")
        shutil.copyfile(custom_config_file, custom_config_file + ".backup")
    print(f"=> Writing config stubs to {custom_config_file}")
    with open(custom_config_file, 'a') as fout:
        fout.writelines(lines)


def generate_configs(system):
    # Get all the benchmark and scenario strings
    for benchmark in Benchmark:
        for scenario in Scenario:
            if os.path.exists(f"configs/{benchmark.valstr()}/{scenario.valstr()}/__init__.py"):
                generate_config(benchmark, scenario, system)


def yes_no_prompt(message, default=True):
    choices = ["", "y", "n"]
    if default is True:
        choice_str = "[Y/n]"
    elif default is False:
        choice_str = "[y/N]"
    elif default is None:
        choice_str = "[y/n]"
        choices = ["y", "n"]
    else:
        raise ValueError(f"Invalid option for default prompt choice: {default}")

    resp = None
    #while resp is None or resp.lower() not in choices:
    #    resp = input(f"{message} {choice_str}: ")

    if resp == "":
        return default
    else:
        return resp == "y"


def main():
    # Inform the user on what this script does.
    print(__doc__)

    # Show the detected system
    print("============= DETECTED SYSTEM ==============\n")
    print(DETECTED_SYSTEM.pretty_string())
    print("\n============================================")

    # Check if the system already matches something
    if MATCHED_SYSTEM is not None:
        s = "!" * 80 + "\n"
        s += (f"WARNING: The system is already a known submission system ({MATCHED_SYSTEM}).\n"
              "You can either quit this script (Ctrl+C) or continue anyway.\n"
              "Continuing will perform the actions described above, and the current system description will be replaced.")
        s += "\n" + "!" * 80
        print(s)

        resp = yes_no_prompt("Continue?")
        if not resp:
            print("Exiting.")
            return

    # Set NUMA Config to None to prevent it matching
    object.__setattr__(DETECTED_SYSTEM, "numa_conf", None)

    custom_list_module = None
    if importlib.util.find_spec(CUSTOM_LIST_MODULE_PATH) is not None:
        print("=> Custom systems list already exists. Reusing.")
        custom_list_module = importlib.import_module(CUSTOM_LIST_MODULE_PATH)
        for existing_sys_id in custom_list_module.custom_systems:
            print(f"  => Discovered existing custom system: {existing_sys_id}")
        # Check that the system isn't in custom_list already
        for k, v in enumerate(custom_list_module.custom_systems):
            if v == DETECTED_SYSTEM:
                print(f"=> System ID '{k}' in existing custom_list.py matches DETECTED_SYSTEM.")
                print(f"=> Exiting.")
                return

        with open(CUSTOM_LIST_FILE_PATH) as f:
            lines = f.readlines()
    else:
        with open(CUSTOM_LIST_STUB_FILE) as f:
            lines = f.readlines()

    print("=> A system ID is a string containing only letters, numbers, and underscores")
    print("=> that is used as the human-readable name of the system. It is also used as")
    print("=> the system name when creating the measurements/ and results/ entries.")
    print("=> This string should also start with a letter to be a valid Python enum member name.")

    # Verify the system name:
    sys_id = ""

    # If DETECTED_SYSTEM already matches an NVIDIA system, its system_id will already be set. Re-use if so.
    if DETECTED_SYSTEM.get_id() is not None:
        sys_id = DETECTED_SYSTEM.get_id()

    while not SYSTEM_NAME_PATTERN.fullmatch(sys_id):
        sys_id = input("=> Specify the system ID to use for the current system: ")

        # Check if the chosen name conflicts with an existing name
        if custom_list_module and sys_id in custom_list_module.custom_systems:
            print(f"=> '{sys_id}' is already being used as the system_id for a different custom system")
            print(f"=> Please enter a different name")
            sys_id = ""

    DETECTED_SYSTEM.set_id(sys_id)
    system_copy = deepcopy(DETECTED_SYSTEM)

    # If the detected system is using a KnownGPU and the detected GPU name is not the primary name of the KnownGPU's
    # AliasedName, __hash__ will return different results. This causes issues with inheritance in configs.
    # If the GPU is a KnownGPU, reset the value.
    # TODO: Heterogeneous systems are not supported yet by the harness or builder
    detected_accelerators = system_copy.accelerator_conf.get_accelerators()
    if len(detected_accelerators) == 1:
        accelerator = detected_accelerators[0]
        for gpu in KnownGPU:
            if accelerator == gpu.value:
                print(f"=> Detected GPU is a known GPU: {gpu.name}")
                new_key = KnownGPUSpoof(accelerator, gpu)
                system_copy.accelerator_conf.layout[new_key] = system_copy.accelerator_conf.layout[accelerator]
                del system_copy.accelerator_conf.layout[accelerator]
                break

    # Add the system to the file
    lines.insert(-5, f"custom_systems['{sys_id}'] = " + obj_to_codestr(system_copy) + "\n")
    with open(CUSTOM_LIST_FILE_PATH, 'w') as f:
        f.writelines(lines)

    # Reload system_list so that the KnownSystem Enum updates with our new system
    reload_system_list()
    print("  => Reloaded system list. MATCHED_SYSTEM:", MATCHED_SYSTEM)

    print("=> This script will generate Benchmark Configuration stubs for the detected system.")
    generate_benchmark_confs = yes_no_prompt("Continue?")
    if not generate_benchmark_confs:
        return

    print(f"=> Generating configs for {sys_id}...")
    generate_configs(DETECTED_SYSTEM)


if __name__ == "__main__":
    main()
