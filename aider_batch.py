import os
import subprocess
import time
import json
from pathlib import Path

start_time = time.time()


jobname = "aider_gemini_bf"
mode = "bugfixing"
instance_list_path = "data/g2_sane_python_instances.txt"
input_data_path = "data/codearena_instances_python.json"
model = "openrouter/google/gemini-2.5-flash"
model_provider = "openrouter"
output_dir = f"logs/baselines/{jobname}"
output_file = f"{output_dir}/all_preds.jsonl"
mem = "32G"

api_key = os.getenv("OMNICODE_API_KEY", None)
if api_key is None:
    raise RuntimeError(f"Could not find OMNICODE_API_KEY environment variable")

all_instances = open(instance_list_path, 'r', encoding='utf-8').read().splitlines()

completed_instances = []
if Path(output_file).exists():
    completed_instances = [
        json.loads(line)['instance_id']
        for line in open(output_file, "r").read().splitlines()
    ]
instances = [i for i in all_instances if i not in completed_instances]
print(f"{len(all_instances)=}, {len(instances)=}")

def pred_instance(instance): 
    print("Processing predict instance:", instance)
    run_id = f"{jobname}_{instance}"
    wrap_cmd = [
        "python baselines/aider/aider_regular.py",
        f"-i {input_data_path}",
        f"-o {output_dir}",
        f"--model_name {model}",
        f"-k {api_key}",
        f"--model_provider {model_provider}",
        f"--mode {mode}",
        f"--instance_ids {instance}",
    ]
    wrap_cmd_str = " ".join(wrap_cmd)
    cmd = [
        "sbatch", f"--job-name={run_id}_pred",
        "--cpus-per-task=2",
        f"--mem={mem}",
        "--gres=gpu:1", 
        "--time=2:00:00",
        f"--output=slurm_logs/%x_%j.out",
        f"--error=slurm_logs/%x_%j.err",
        f'--wrap="{wrap_cmd_str}"'
    ]    
    result = subprocess.run(" ".join(cmd), shell=True, text=True,
                        capture_output=True)
    if result.returncode != 0:
        print("sbatch failed:", result.stderr.strip())
    else:
        print(f"{instance}: {result.stdout.strip()}")


for instance in instances:
    # 10 jobs + header + empty line
    while len(subprocess.run(["squeue"], capture_output=True, text=True).stdout.split("\n")) == 22:
        time.sleep(180)
    pred_instance(instance)
    print(f"🔄 Starting predict instance {instance}")


while len(subprocess.run(["squeue"], capture_output=True, text=True).stdout.split("\n")) != 2: # header + empty line
    time.sleep(30)
print("Predict done.")
cost = time.time() - start_time
print(f"Total time: {cost/60:.2f} minutes")

# def eval_instance(instance): 
#     print("Processing evaluate instance:", instance)
#     run_id = instance.split('__')[1]
#     cmd = [
#         "sbatch", f"--job-name={run_id}_eval",
#         "--cpus-per-task=2",
#         f"--mem={mem}",
#         "--gres=gpu=1", 
#         "--time=2:00:00",
#         f"--output=slurm_logs/end2end_java/%x_%j.out",
#         f"--error=slurm_logs/end2end_java/%x_%j.err",
#         f'--wrap="python codearena.py --MSWEBugFixing --predictions_path {output_file} --run_id {run_id} --max_workers 1 --mswe_phase all --force_rebuild True --clean True --use_apptainer True --instance_ids {instance} --timeout 10000 --g2 True"'
#     ]    
#     cmd_str = " ".join(cmd)
#     ret = os.system(cmd_str)

#     if ret >> 8 != 0:
#         print(f"Error processing instance {instance}: {ret >> 8}")
#     else:
#         print(f"Successfully processed instance {instance}")

# for instance in instances:
#     while len(subprocess.run(["squeue"], capture_output=True, text=True).stdout.split("\n")) == 7:
#         time.sleep(180)
#     eval_instance(instance)
#     print(f"🔄 Starting evaluate instance {instance}")
#     # 40 jobs + header + empty line

# while len(subprocess.run(["squeue"], capture_output=True, text=True).stdout.split("\n")) != 2: # header + empty line
#     time.sleep(30)
# print("Evaluate done.")

# cost = time.time() - start_time
# print(f"Total time: {cost/60:.2f} minutes")