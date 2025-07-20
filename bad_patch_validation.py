import codearena
import os
import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Verify Bad Patch")
parser.add_argument( "--instance_id", help="Instance ID")
parser.add_argument("--results_folder", help="run evaluation folder name (after logs/run_evaluation/)")
# parser.add_argument("--dataset_name", default="data/codearena_instances.json",
#                     help="Name of the dataset")
# parser.add_argument("--dataset_name", default="data/agentless_instances.json",
#                     help="Name of the dataset")
parser.add_argument("--dataset_name", default="data/java_instances.json",
                    help="Name of the dataset")
parser.add_argument("--language", default="python")
args = parser.parse_args()

if args.language == 'python':
    results_dir = os.path.join('logs/run_evaluation', args.results_folder)
elif args.language == 'java':
    results_dir = os.path.join('multiswebench_runs/BugFixing')

if not os.path.exists(results_dir):
    print('Results folder does not exist (likely due to patch being empty)', results_dir)
    sys.exit(1)

if args.language == 'python':
    full_dir = os.path.join(results_dir, 'agentless', args.instance_id)
    work_dir = full_dir
    report_path = os.path.join(full_dir, 'report.json')
elif args.language == 'java':
    full_dir = os.path.join(results_dir, 'output', f"run_{args.results_folder}")
    work_dir_temp = os.path.join(results_dir, 'workdir', f"run_{args.results_folder}")
    for root, dirs, files in os.walk(work_dir_temp):
        if 'fix.patch' in files:
            work_dir = root
            break
    report_path = os.path.join(full_dir, 'final_report.json')

if not os.path.exists(report_path):
    print('Report file does not exist (likely due to error in building image)', report_path)
    sys.exit(1)

with open(report_path, 'r') as f:
    report = json.load(f)

if args.language == 'python':
    unresolved = report[args.instance_id]['unresolved']
elif args.language == 'java':
    unresolved = report['unresolved_instances'] > 0
if not unresolved:
    print('Solved task or Error:', results_dir)
    sys.exit(1)
else:

    with open(args.dataset_name, 'r') as f:
        dataset = json.load(f)

    # load the patch from predictions path
    if args.language == 'python':
        patch_path = os.path.join(work_dir, 'patch.diff')
    elif args.language == 'java':
        patch_path = os.path.join(work_dir, 'fix.patch')
    with open(patch_path, 'r') as f:
        patch = f.read()

    # add patch to dataset
    try:
        task_ix = [i for i in range(len(dataset)) if dataset[i]['instance_id'] == args.instance_id][0]
    except:
        print("!!! instance not in output file already")
    
    if 'bad_patches' in dataset[task_ix]:
        index = len(dataset[task_ix]['bad_patches']) + 1
        dataset[task_ix]['bad_patches'].append({
            "idx": index,
            "source": "agentless",
            "patch": patch
            })
    else:
        dataset[task_ix]['bad_patches'] = [patch]
    # dataset[task_ix]['bad_patch'] = patch

    # save dataset back to json file
    with open(args.dataset_name, 'w') as f:
        json.dump(dataset, f, indent=4)

    # save gold patch to gold.diff for easy comparison
    gold_path = os.path.join(work_dir, 'gold.diff')
    with open(gold_path, 'w+') as f:
        f.write(dataset[task_ix]['patch'])

    print('Bad patch successfully added to dataset from:', results_dir)

    found_bad_patch = True
    sys.exit(0)

