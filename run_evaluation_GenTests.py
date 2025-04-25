from __future__ import annotations

import docker
import json
import resource
import traceback

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_utils import (
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    cleanup_container,
    list_images,
    should_remove,
    clean_images,
)
from docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from CodeArena_grading import get_eval_report_test_generation, get_fail_to_fail
#from swebench.swebench.harness.test_spec import make_test_spec, TestSpec
from CodeArena_test_spec import make_test_spec, TestSpec
from swebench.harness.utils import str2bool
from utils import load_swebench_dataset, load_CodeArena_prediction_dataset, update_test_spec_with_specific_test_names

class EvaluationError(Exception):
    def __init__(self, instance_id, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.instance_id = instance_id
        self.log_file = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Evaluation error for {self.instance_id}: {self.super_str}\n"
            f"Check ({self.log_file}) for more information."
        )

def get_gold_predictions(dataset_name: str, instance_ids: list, split: str):
    """
    Get ground truth tests and their corresponding patches from the FAIL_TO_PASS section.
    
    Args:
        dataset_name (str): Name of the dataset
        instance_ids (list): List of instance IDs to process
        split (str): Dataset split to use
        
    Returns:
        list: List of dictionaries containing instance IDs, patches, and failing test information
    """
    dataset = load_swebench_dataset(dataset_name, split)
    results = []
    
    for datum in dataset:
        if datum[KEY_INSTANCE_ID] not in instance_ids:
            continue
        
        # loading gold prediction results assumes direct employment of swe-bench-verified
        result = {
            KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
            "repo": datum["repo"],
            "base_commit": datum["base_commit"],
            "gold_patch": datum["patch"],
            "candidate_test_patch": datum["test_patch"], # gold test patch
            "version": datum["version"],
            "model_name_or_path": "gold"
        }
        
        # Add bad patches if they exist
        if "bad_patches" in datum:
            result["bad_patches"] = datum["bad_patches"]
        elif "bad_patch" in datum:
            result["bad_patches"] = [datum["bad_patch"]]
            
        results.append(result)
    
    return results

def get_dataset_from_preds(
    dataset_name: str,
    split: str,
    instance_ids: list,
    run_id: str,
    exclude_completed: bool = True,
    codearena_instances: str = "data/codearena_instances.json",
    generated_tests_path: str = "generated_tests.jsonl"
):
    """
    Return only instances that have predictions and are in the dataset as a list of dicts.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # Process the SWE-Bench dataset and merge with generated tests and bad patches
    merged_df = load_CodeArena_prediction_dataset(generated_tests_path, codearena_instances, instance_ids)

    # Now extract the merged instances from the DataFrame into the dataset
    dataset_ids = set(merged_df['instance_id'])

    if instance_ids:
        # Ensure only requested instance_ids are considered
        missing_preds = set(instance_ids) - dataset_ids
        if missing_preds:
            print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs.")
        
        # Filter merged_df to only include requested instance_ids
        merged_df = merged_df[merged_df['instance_id'].isin(instance_ids)]
    
    # Check which instance IDs have already been run
    completed_ids = set()
    for _, instance in merged_df.iterrows():
        prediction = merged_df[merged_df['instance_id'] == instance['instance_id']].iloc[0]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / (prediction['instance_id']+"_testGeneration")
            / "report.json"
        )
        if report_file.exists():
            completed_ids.add(instance['instance_id'])

    if completed_ids and exclude_completed:
        # Filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        merged_df = merged_df[~merged_df['instance_id'].isin(completed_ids)]

    # Filter dataset to only instances with predictions and non-empty patches
    merged_df = merged_df[merged_df['instance_id'].isin(dataset_ids)]

    print(merged_df.to_dict(orient="records")[0].keys())

    # Convert the DataFrame to a list of dictionaries
    return merged_df.to_dict(orient="records")



def make_run_report(
        predictions: dict,
        full_dataset: list,
        client: docker.DockerClient,
        run_id: str
    ) -> Path:
    """
    Make a final evaluation and run report of the instances that have been run.
    Also reports on images and containers that may still running!

    Args:
        predictions (dict): Predictions dict generated by the model
        full_dataset (list): List of all instances
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
    
    Returns:
        Path to report file
    """
    # instantiate sets to store IDs of different outcomes
    completed_ids = set()
    successful_ids = set()
    error_ids = set()
    unstopped_containers = set()
    unremoved_images = set()
    unsuccessful_ids = set()
    incomplete_ids = set()
    # get instances with empty patches
    empty_patch_ids = set()

    # iterate through dataset and check if the instance has been run
    for instance in full_dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id not in predictions:
            # skip instances without 
            incomplete_ids.add(instance_id)
            continue
        prediction = predictions[instance_id]
        # TODO: Make this look nicer. External predictions do not conform to candidate_test_patch format
        if prediction.get("candidate_test_patch", None) in ["", None] and prediction.get("model_patch", None) in ["", None]:
            empty_patch_ids.add(instance_id)
            continue
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / (prediction[KEY_INSTANCE_ID]+"_testGeneration")
            / "report.json"
        )
        if report_file.exists():
            # If report file exists, then the instance has been run
            completed_ids.add(instance_id)
            report = json.loads(report_file.read_text())
            if report[instance_id]["Test_Accept"]:
                # Record if the instance was resolved
                successful_ids.add(instance_id)
            else:
                unsuccessful_ids.add(instance_id)
        else:
            # Otherwise, the instance was not run successfully
            error_ids.add(instance_id)

    # get remaining images and containers
    images = list_images(client)
    test_specs = list(map(make_test_spec, full_dataset))
    for spec in test_specs:
        image_name = spec.instance_image_key
        if image_name in images:
            unremoved_images.add(image_name)
    containers = client.containers.list(all=True)
    for container in containers:
        if run_id in container.name:
            unstopped_containers.add(container.name)

    # print final report
    dataset_ids = {i[KEY_INSTANCE_ID] for i in full_dataset}
    print(f"Total instances: {len(full_dataset)}")
    print(f"Instances submitted: {len(set(predictions.keys()) & dataset_ids)}")
    print(f"Instances completed: {len(completed_ids)}")
    print(f"Instances incomplete: {len(incomplete_ids)}")
    print(f"Test Case Accepted: {len(successful_ids)}")
    print(f"Test Case Rejected: {len(unsuccessful_ids)}")
    print(f"Instances with empty test patches: {len(empty_patch_ids)}")
    print(f"Instances with errors: {len(error_ids)}")
    print(f"Unstopped containers: {len(unstopped_containers)}")
    print(f"Unremoved images: {len(unremoved_images)}")

    # write report to file
    report = {
        "total_instances": len(full_dataset),
        "submitted_instances": len(predictions),
        "completed_instances": len(completed_ids),
        "Test Case Accepted": len(successful_ids),
        "Test Case Rejected": len(unsuccessful_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(predictions.keys())),
        "tests_accepted": list(sorted(successful_ids)),
        "tests_rejected": list(sorted(unsuccessful_ids)),
        "error_ids": list(sorted(error_ids)),
        "unstopped_containers": list(sorted(unstopped_containers)),
        "unremoved_images": list(sorted(unremoved_images)),
        "schema_version": 2,
    }
    #report_file = Path(
    #    list(predictions.values())[0]["model_name_or_path"].replace("/", "__")
    #    + f".{run_id}"
    #    + ".json"
    #)
    #with open(report_file, "w") as f:
    #    print(json.dumps(report, indent=4), file=f)
    #print(f"Report written to {report_file}")
    return report_file

def main(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions_path: str,
        max_workers: int,
        force_rebuild: bool,
        cache_level: str,
        clean: bool,
        open_file_limit: int,
        run_id: str,
        timeout: int,
    ):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    # load predictions as map of instance_id to prediction
    if predictions_path == 'gold':
        print("Using gold predictions - ignoring predictions_path")
        predictions = get_gold_predictions(dataset_name, instance_ids, split) # Gold Prediction should correspond to ground truth test (PASS TO FAIl)
    else:
        if predictions_path.endswith(".json"):
            with open(predictions_path, "r") as f:
                predictions = json.load(f)
        elif predictions_path.endswith(".jsonl"):
            with open(predictions_path, "r") as f:
                predictions = [json.loads(line) for line in f]
        else:
            raise ValueError("Predictions path must be \"gold\", .json, or .jsonl")
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    if(not predictions_path == 'gold'):
        dataset = get_dataset_from_preds(dataset_name, split, instance_ids, run_id=run_id, generated_tests_path=predictions_path)
    else:
        dataset = get_gold_predictions(dataset_name, instance_ids, split)
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids, full=True)
    existing_images = list_images(client)
    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:
        print("No instances to run.")
    else:
        # build environment images + run instances
        build_env_images(client, dataset, force_rebuild, max_workers)
        run_instances(predictions, dataset, cache_level, clean, force_rebuild, max_workers, run_id, timeout)

    # clean images + make final report
    clean_images(client, existing_images, cache_level, clean)
    make_run_report(predictions, full_dataset, client, run_id)

def run_instance(
        test_spec: TestSpec,
        pred: dict,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        run_id: str,
        timeout: int | None = None,
    ):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / (instance_id+"_testGeneration")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Link the image build dir in the log dir
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except:
            pass
    log_file = log_dir / "run_instance.log"

    # Set up report file + logger
    report_path = log_dir / "report.json"
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    container = None
    try:
        # Build + start instance container (instance image should already be built)
        container = build_container(test_spec, client, run_id, logger, rm_image, force_rebuild)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        update_test_spec_with_specific_test_names(test_spec=test_spec, repo_path=Path("/testbed"))

        # TODO: Before candidate test patch is applied, determine F2F tests.
        ######################################################################

        # Step 1: Run F2F check with gold patch
        test_output_path_f2f = log_dir / "f2f_check.txt"
        eval_file = Path(log_dir / "gold_eval.sh")
        eval_file.write_text(test_spec.inverted_eval_script_gold)
        logger.info(f"Eval script for Gold Evaluation written to {eval_file}")
        copy_to_container(container, eval_file, Path("/gold_eval.sh"))

        # Run F2F check
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /gold_eval.sh", timeout)
        logger.info(f'F2F check runtime: {total_runtime:_.2f} seconds')
        with open(test_output_path_f2f, "w") as f:
            f.write(test_output)
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(instance_id, f"Test timed out after {timeout} seconds.", logger)

        # Get F2F tests
        fail_to_fail = get_fail_to_fail(test_output_path_f2f)

        # Step 2: Apply and test candidate test patch
        # Get initial git diff before applying test patch
        git_diff_before_test = container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
        logger.info(f"Git diff before test patch:\n{git_diff_before_test}")

        patch_file = Path(log_dir / "patch.diff")
        patch_content = pred.get("candidate_test_patch") or pred.get("model_patch") or ""
        patch_file.write_text(patch_content)
        logger.info(f"Candidate Test Patch written to {patch_file}")
        copy_to_container(container, patch_file, Path("/tmp/patch.diff"))

        # Apply patch
        val = container.exec_run("git apply --allow-empty -v /tmp/patch.diff", workdir="/testbed", user="root")
        if val.exit_code != 0:
            logger.info("First patch attempt failed, trying with more permissive options...")
            val = container.exec_run("patch --batch --fuzz=5 -p1 -i /tmp/patch.diff", workdir="/testbed", user="root")
            if val.exit_code != 0:
                logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}")
                raise EvaluationError(instance_id, f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}", logger)
            else:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")
        else:
            logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")

        # Get git diff after applying test patch
        git_diff_after_test = container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
        logger.info(f"Git diff after test patch:\n{git_diff_after_test}")

        # Step 3: Run gold patch evaluation
        # Get git diff before running gold eval
        git_diff_before_gold = container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
        logger.info(f"Git diff before gold evaluation:\n{git_diff_before_gold}")

        test_output_path_gold = log_dir / "gold_test_output.txt"
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /gold_eval.sh", timeout)
        logger.info(f'Gold evaluation runtime: {total_runtime:_.2f} seconds')
        
        # Get git diff after running gold eval
        git_diff_after_gold = container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
        logger.info(f"Git diff after gold evaluation:\n{git_diff_after_gold}")

        with open(test_output_path_gold, "w") as f:
            f.write(test_output)
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(instance_id, f"Test timed out after {timeout} seconds.", logger)

        # Reset to clean state before bad patches
        reset_val = container.exec_run("git reset --hard HEAD", workdir="/testbed", user="root")
        if reset_val.exit_code != 0:
            logger.warning("Failed to reset before bad patches")
            reset_val = container.exec_run("git checkout -- .", workdir="/testbed", user="root")
            if reset_val.exit_code != 0:
                logger.error("All reset attempts failed before bad patches")
                raise EvaluationError(instance_id, "Failed to reset before bad patches", logger)

        # Verify clean state after reset
        git_diff_after_reset = container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
        if git_diff_after_reset:
            logger.warning(f"Codebase not clean after reset before bad patches:\n{git_diff_after_reset}")

        # Step 4: Run bad patch evaluations
        test_output_paths_bad = []
        bad_patches = []
        
        # Get all bad patches
        if 'bad_patches' in pred:
            bad_patches = pred['bad_patches']
        elif 'bad_patch' in pred:
            bad_patches = [pred['bad_patch']]

        # Create the base bad eval script
        base_eval_script = test_spec.inverted_eval_script_bad

        # Create and copy eval script for each bad patch
        for i, bad_patch in enumerate(bad_patches):
            # Get initial git diff before applying bad patch
            git_diff_before = container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
            logger.info(f"Git diff before bad patch {i}:\n{git_diff_before}")

            # Create a new eval script for this specific bad patch
            eval_script = base_eval_script.replace(
                f"EOF_{test_spec.instance_id}",
                f"EOF_{test_spec.instance_id}_bad_{i}"
            )
            eval_script = eval_script.replace(
                test_spec.bad_patches[0],  # Replace the first bad patch
                bad_patch  # With the current bad patch
            )
            
            eval_file = Path(log_dir / f"bad_eval_{i}.sh")
            eval_file.write_text(eval_script)
            logger.info(f"Bad patch {i} evaluation script written to {eval_file}")
            copy_to_container(container, eval_file, Path(f"/bad_eval_{i}.sh"))

            # Run evaluation for this bad patch
            test_output_path_bad = log_dir / f"bad_test_output_{i}.txt"
            test_output_paths_bad.append(test_output_path_bad)
            
            test_output, timed_out, total_runtime = exec_run_with_timeout(container, f"/bin/bash /bad_eval_{i}.sh", timeout)
            logger.info(f'Bad patch {i} evaluation runtime: {total_runtime:_.2f} seconds')
            
            # Get git diff after running the evaluation
            git_diff_after = container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
            logger.info(f"Git diff after bad patch {i}:\n{git_diff_after}")
            
            with open(test_output_path_bad, "w") as f:
                f.write(test_output)
                if timed_out:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    logger.warning(f"Bad patch {i} evaluation timed out after {timeout} seconds")
                    continue

            # Reset to clean state for next patch
            reset_val = container.exec_run("git reset --hard HEAD", workdir="/testbed", user="root")
            if reset_val.exit_code != 0:
                logger.warning(f"Failed to reset after bad patch {i}")
                # Try alternative reset method
                reset_val = container.exec_run("git checkout -- .", workdir="/testbed", user="root")
                if reset_val.exit_code != 0:
                    logger.error(f"All reset attempts failed for bad patch {i}")
                    continue

            # Verify clean state after reset
            git_diff_after_reset = container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
            if git_diff_after_reset:
                logger.warning(f"Codebase not clean after reset for bad patch {i}:\n{git_diff_after_reset}")

        # Step 5: Generate final report
        logger.info("Generating evaluation report...")
        if test_output_paths_bad:
            report = get_eval_report_test_generation(
                test_spec=test_spec,
                prediction=pred,
                log_paths=[test_output_path_gold] + test_output_paths_bad,
                include_tests_status=True,
                fail_to_fail_tests=fail_to_fail
            )
            logger.info(f"Result: Test Accepted: {report[instance_id]['Test_Accept']}")
        else:
            logger.warning("No bad patch evaluations completed successfully")
            report = {
                instance_id: {
                    "patch_is_None": False,
                    "patch_exists": True,
                    "gold_patch_successfully_applied": True,
                    "Test_Accept": False,
                    "error": "No bad patch evaluations completed successfully"
                }
            }

        # Save report
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report

    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
    finally:
        if container:
            cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return


def run_instances(
        predictions: dict,
        instances: list,
        cache_level: str,
        clean: bool,
        force_rebuild: bool,
        max_workers: int,
        run_id: str,
        timeout: int,
    ):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    client = docker.from_env()
    test_specs = list(map(make_test_spec, instances)) # will include inverted evaluation script for gold and bad patch

    # print number of existing instance images
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    # run instances in parallel
    print(f"Running {len(instances)} instances...")
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
                    # TODO: Either optimize this lookup or find a more elegant way to pass corrected predictions
                    next((item for item in instances if item["instance_id"] == test_spec.instance_id), None),
                    should_remove(
                        test_spec.instance_image_key,
                        cache_level,
                        clean,
                        existing_images,
                    ),
                    force_rebuild,
                    client,
                    run_id,
                    timeout,
                ): None
                for test_spec in test_specs
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if instance ran successfully
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    continue
    print("All instances run.")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Verified", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers (should be <= 75%% of CPU cores)")
    parser.add_argument("--open_file_limit", type=int, default=4096, help="Open file limit")
    parser.add_argument(
        "--timeout", type=int, default=1_800, help="Timeout (in seconds) for running tests for each instance"
        )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images"
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument("--run_id", type=str, required=True, help="Run ID - identifies the run")
    args = parser.parse_args()

    main(**vars(args))
