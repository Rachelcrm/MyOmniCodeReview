#!/usr/bin/env bash

set -euo pipefail


INSTANCE_FILE="data/multiswebench_data/mswebench_instance_ids_original.txt"
RUN_ID="gold_java_bf_check"

LOG_DIR="slurm_logs/${RUN_ID}"

CPUS=4 
MEM=8192         # MiB 
TIME_LIMIT="02:00:00"

mkdir -p "${LOG_DIR}"

while IFS= read -r ID || [[ -n "${ID}" ]]; do
    SAN_ID="${ID//\//__}"      # 1)  /  →  __
    SAN_ID="${SAN_ID//:/_}"    # 2)  :  →  _
    JOB_NAME="${RUN_ID}_${SAN_ID}"

    echo "Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})"

    sbatch --job-name="${JOB_NAME}" \
           --cpus-per-task="${CPUS}" \
           --mem="${MEM}" \
           --time="${TIME_LIMIT}" \
           --output="${LOG_DIR}/%x_%j.out" \
           --error="${LOG_DIR}/%x_%j.err" \
           --wrap="python codearena.py --MSWEBugFixing \
                  --predictions_path gold \
                  --run_id ${JOB_NAME} \
                  --max_workers 1 \
                  --mswe_phase all \
                  --force_rebuild True \
                  --clean True \
                  --use_apptainer True \
                  --instance_ids ${ID}"
done < "${INSTANCE_FILE}"