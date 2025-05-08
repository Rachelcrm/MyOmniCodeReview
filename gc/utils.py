import json
import time

from google.cloud import compute_v1
from google.api_core.exceptions import NotFound


def list_to_bash_array(data):
    """
    Converts a Python list to a string representation of a Bash array.

    Args:
        data: A Python list.

    Returns:
        A string representing the list as a Bash array.
    """
    return " ".join(map(json.dumps, data))

def create_image_from_vm(
    project_id: str,
    zone: str,
    source_instance: str,
    image_name: str,
    image_family: str = "batch-jobs",
    image_description: str = "Image created for batch processing",
    timeout_seconds: int = 600,
):
    """
    Creates a new VM image from an existing instance.
    
    Args:
        project_id: The GCP project ID.
        zone: The zone where the source instance is located.
        source_instance: The name of the source instance to create the image from.
        image_name: The name for the new image.
        image_family: Optional image family to group related images.
        image_description: Optional description for the image.
        timeout_seconds: Maximum time to wait for the operation to complete.
        
    Returns:
        The created image object.
    """
    # Initialize the Compute Engine client
    instance_client = compute_v1.InstancesClient()
    image_client = compute_v1.ImagesClient()
    
    # Get the source instance
    instance = instance_client.get(
        project=project_id, 
        zone=zone, 
        instance=source_instance
    )
    
    # Create the image from the instance
    image = compute_v1.Image()
    image.name = image_name
    image.source_disk = f"projects/{project_id}/zones/{zone}/disks/{source_instance}"
    image.family = image_family
    image.description = image_description
    
    # Start the image creation operation
    print(f"Creating image '{image_name}' from instance '{source_instance}'...")
    operation = image_client.insert(project=project_id, image_resource=image)
    
    # Wait for the operation to complete
    start_time = time.time()
    operation_client = compute_v1.GlobalOperationsClient()
    
    while True:
        # Check if we've exceeded the timeout
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Image creation timed out after {timeout_seconds} seconds")
        
        # Get the operation status
        operation_status = operation_client.get(project=project_id, operation=operation.name)
        
        # Check if the operation is done
        if operation_status.status == compute_v1.Operation.Status.DONE:
            break
        
        # Print status and wait before checking again
        print(f"Image creation in progress... Status: {operation_status.status.name}")
        time.sleep(10)
    
    # Check if the operation was successful
    if operation_status.error:
        error_messages = [error.message for error in operation_status.error.errors]
        raise Exception(f"Image creation failed: {error_messages}")
    
    # Get and return the created image
    created_image = image_client.get(project=project_id, image=image_name)
    print(f"Image '{image_name}' created successfully with ID: {created_image.id}")
    
    return created_image

def get_latest_image_from_family(project_id: str, family: str):

    """
    Retrieves the latest image from a specified image family.
    
    Args:
        project_id: The GCP project ID.
        family: The image family to search in.
        
    Returns:
        The URL of the latest image in the family, or None if not found.
    """
    image_client = compute_v1.ImagesClient()
    
    try:
        # Get the latest image from the family
        image = image_client.get_from_family(project=project_id, family=family)
        image_url = f"projects/{project_id}/global/images/{image.name}"
        print(f"Found latest image '{image.name}' from family '{family}'")
        return image_url
    except Exception as e:
        print(f"Error retrieving latest image from family '{family}': {e}")
        return None


def wait_for_operation(operation, project_id, zone):
    """Wait for a compute engine zone operation to complete."""
    print(f"Waiting for operation {operation.name} to complete...")
    
    # Create a zone operations client
    operation_client = compute_v1.ZoneOperationsClient()
    
    while operation.status != compute_v1.Operation.Status.DONE:
        time.sleep(5)
        operation = operation_client.get(
            project=project_id, 
            zone=zone,
            operation=operation.name
        )
    
    if operation.error:
        print(f"Error during operation: {operation.error}")
        return False
    return True


def check_vm_exists(instance_client, project_id, zone, vm_name):
    """Check if a VM with the given name exists."""
    try:
        instance_client.get(project=project_id, zone=zone, instance=vm_name)
        return True
    except NotFound:
        return False


def get_vm_status(instance_client, project_id, zone, vm_name):
    """Get the status of a VM."""
    instance = instance_client.get(project=project_id, zone=zone, instance=vm_name)
    return instance.status


def start_vm(instance_client, project_id, zone, vm_name):
    """Start a stopped VM."""
    print(f"Starting existing VM {vm_name}...")
    operation = instance_client.start(project=project_id, zone=zone, instance=vm_name)
    return operation

def delete_vm(instance_client, project_id, zone, vm_name):
    """Delete a Compute Engine VM."""
    try:
        print(f"Deleting VM {vm_name}...")
        operation = instance_client.delete(
            project=project_id,
            zone=zone,
            instance=vm_name
        )
        wait_result = wait_for_operation(operation, project_id, zone)
        if wait_result:
            print(f"Successfully deleted VM: {vm_name}")
        return wait_result
    except Exception as e:
        print(f"Error deleting VM {vm_name}: {e}")
        return False



def reset_vm(instance_client, project_id, zone, vm_name, startup_script, instance_ids):
    """Reset a VM by updating its metadata with new startup script and instance IDs."""
    print(f"Updating metadata for VM {vm_name}...")
    
    # Get the current instance
    instance = instance_client.get(project=project_id, zone=zone, instance=vm_name)
    
    # Create a new metadata object with the updated startup script
    metadata = compute_v1.Metadata()
    
    # Copy existing metadata items if they exist
    existing_items = []
    if instance.metadata and instance.metadata.items:
        for item in instance.metadata.items:
            if item.key != "startup-script":
                existing_items.append(item)
    
    # Add the new startup script item
    script_item = compute_v1.Items()
    script_item.key = "startup-script"
    script_item.value = startup_script
    existing_items.append(script_item)
    
    # Set all items
    metadata.items = existing_items
    
    # Set the fingerprint to ensure we're updating the latest version
    if instance.metadata:
        metadata.fingerprint = instance.metadata.fingerprint
    
    # Update the metadata
    operation = instance_client.set_metadata(
        project=project_id,
        zone=zone,
        instance=vm_name,
        metadata_resource=metadata
    )
    
    return operation

def list_images_in_family(project_id: str, family: str):
    """
    Lists all images in a specified image family.
    
    Args:
        project_id: The GCP project ID.
        family: The image family to search in.
        
    Returns:
        A list of image objects in the family.
    """
    image_client = compute_v1.ImagesClient()
    
    # Filter by family
    filter_str = f'family="{family}"'
    
    try:
        # List all images matching the filter
        images = list(image_client.list(project=project_id, filter=filter_str))
        print(f"Found {len(images)} images in family '{family}'")
        return images
    except Exception as e:
        print(f"Error listing images in family '{family}': {e}")
        return []

def delete_image(project_id: str, image_name: str):
    """
    Deletes a specified image.
    
    Args:
        project_id: The GCP project ID.
        image_name: The name of the image to delete.
        
    Returns:
        True if deletion was successful, False otherwise.
    """
    image_client = compute_v1.ImagesClient()
    operation_client = compute_v1.GlobalOperationsClient()
    
    try:
        # Delete the image
        print(f"Deleting image '{image_name}'...")
        operation = image_client.delete(project=project_id, image=image_name)
        
        # Wait for the operation to complete
        while operation.status != compute_v1.Operation.Status.DONE:
            time.sleep(5)
            operation = operation_client.get(
                project=project_id,
                operation=operation.name
            )
        
        if operation.error:
            print(f"Error deleting image '{image_name}': {operation.error}")
            return False
        
        print(f"Image '{image_name}' deleted successfully")
        return True
    except Exception as e:
        print(f"Error deleting image '{image_name}': {e}")
        return False


def check_image_exists(
    project_id: str,
    image_name: str,
):
    image_client = compute_v1.ImagesClient()

    image_exists = False
    request = compute_v1.ListImagesRequest(project=project_id)
    for image in image_client.list(request=request):
        if image.name == image_name:
            image_exists = True
            print(f"Existing image found: {image_name}")
            break

    return image_exists

def create_image_wrapped(
    rebuild: bool,
    project_id: str,
    zone: str,
    source_instance: str,
    timeout_seconds: int = 600,
):
    """
    Wrapper around `create_image_for_vm` that checks if image already exists
    
    Args:
        rebuild: whether to rebuild the image if it already exists
        project_id: The GCP project ID.
        zone: The zone where the source instance is located.
        source_instance: The name of the source instance to create the image from.
        timeout_seconds: Maximum time to wait for the operation to complete.
        
    Returns:
        The created image object.
    """
    # Initialize the image client
    
    # Use static naming based on job_type
    image_name = f"{source_instance}-image"
    
    image_exists = check_image_exists(project_id, image_name)
    
    if rebuild:
        if image_exists:
            delete_image(
                project_id=project_id,
                image_name=image_name,
            )

        image = create_image_from_vm(
            project_id=project_id,
            zone=zone,
            source_instance=source_instance,
            image_name=image_name,
            image_description=f"Image created from {source_instance}",
            timeout_seconds=timeout_seconds,
        )

    else:
        if not image_exists:
            print(f"Could not find image with name: {image_name}")
            return None

    disk_image = f"projects/{project_id}/global/images/{image_name}"
    print(f"Using image: {disk_image}")

    return disk_image

SANITY_CMD = """python codearena.py \
--BugFixing \
--predictions_path gold \
--run_id sanity \
--instance_ids $INSTANCE_ID"""


BAD_PATCH_GEN_CMD = """python baselines/badpatchllm/generate_bad.py \
-o logs/ \
-m gemini-2.5-flash-preview-4-17 \
--run_id bp_gen \
-n 3 \
-d data/codearena_instances.json \
--max_workers 4 \
--instance_ids $INSTANCE_ID"""



AGENTLESS_CHECK_CMD_OLD = """process_files() {
    # Create a temporary directory for logs
    mkdir -p tmp_logs

    # Get all files matching the pattern from Google Storage
    echo "Fetching files from Google Storage..."
    gsutil ls gs://${BUCKET_NAME}/agentless_bad_patches/${INSTANCE_ID}*.jsonl > file_list.txt

    # Check if any files were found
    if [ ! -s file_list.txt ]; then
        echo "No files found for $INSTANCE_ID"
        exit 1
    fi

    # Process each file
    while read file_path; do
        # Extract filename from path
        filename=$(basename "$file_path")
        echo "Processing $filename"
        
        # Download the file
        gsutil cp "$file_path" .
        
        # Run the Python script
        echo "Running codearena.py for $filename"
        python codearena.py \
            --BugFixing \
            --predictions_path "$filename" \
            --run_id agentless_check \
            --instance_ids "$INSTANCE_ID"
        
        # Extract the suffix from filename (e.g., fl_5)
        suffix=$(echo "$filename" | sed "s/${INSTANCE_ID}_//")
        suffix=${suffix%.jsonl}
        
        # Copy logs to tmp_logs with appropriate name
        echo "Copying logs to tmp_logs/${INSTANCE_ID}_${suffix}"
        cp -r logs "tmp_logs/${INSTANCE_ID}_${suffix}"
        
    done < file_list.txt

    # Replace logs with tmp_logs
    echo "Replacing logs directory with tmp_logs"
    rm -rf logs
    mv tmp_logs logs
}

# Execute the function and capture all output
process_files"""


AGENTLESS_CHECK_CMD = """process_files() {
    # Create a temporary directory for logs
    mkdir -p tmp_logs
    
    # Base Google Storage path
    BASE_PATH="gs://${BUCKET_NAME}/agentless_bad_patches"
    
    for i in {0..2}; do
        # Construct the filename
        filename="${INSTANCE_ID}_${i}.jsonl"
        file_path="${BASE_PATH}/${filename}"
        
        # Check if the file exists in Google Storage
        if gsutil -q stat "$file_path"; then
            echo "File $filename exists, processing..."
            
            # Download the file
            gsutil cp "$file_path" .
            
            # Run the Python script
            echo "Running codearena.py for $filename"
            python codearena.py \
                --BugFixing \
                --predictions_path "$filename" \
                --run_id agentless_check \
                --instance_ids "$INSTANCE_ID"
            
            # Copy logs to tmp_logs with appropriate name
            echo "Copying logs to tmp_logs/${INSTANCE_ID}_${i}"
            cp -r logs "tmp_logs/${INSTANCE_ID}_${i}"
        else
            echo "File $filename does not exist, skipping."
        fi
    done
    
    # Replace logs with tmp_logs if any files were processed
    if [ -d "tmp_logs" ] && [ "$(ls -A tmp_logs)" ]; then
        echo "Replacing logs directory with tmp_logs"
        rm -rf logs
        mv tmp_logs logs
    else
        echo "No files were processed, logs directory unchanged."
        rm -rf tmp_logs
    fi
}

process_files"""


PATCH_CHECK_CMD = """process_files() {{
    gcs_path="gs://${{BUCKET_NAME}}/{results_dir}"
    temp_dir=$(mktemp -d)
    
    # Initialize an array to store matching file data
    declare -a founds=()
    
    # Use wildcards to search specifically in directories with INSTANCE_ID prefix
    # This significantly narrows down our search space
    matching_files=$(gsutil ls "${{gcs_path}}/${{INSTANCE_ID}}*/**/all_preds.jsonl")
    
    # Process each potential matching file
    for filepath in $matching_files; do
        # Download the file to temp directory
        local temp_file="${{temp_dir}}/$(basename "$filepath")"
        gsutil cp "${{filepath}}" "${{temp_file}}"
        
        # Process the file to find matching instance_id
        while IFS= read -r line; do
            if [ "$(echo "$line" | jq -r '.instance_id // empty')" = "${{INSTANCE_ID}}" ]; then
                founds+=("$line")
                break  # Found a match in this file
            fi
        done < "${{temp_file}}"
        
        # Remove temp file after processing
        rm "${{temp_file}}"
    done
    
    # Check if no predictions were found
    if [ ${{#founds[@]}} -eq 0 ]; then
        echo "No predictions found for ${{INSTANCE_ID}}, skipping ..."
        # Clean up temp directory
        rm -rf "${{temp_dir}}"
        return 0
    fi
    
    # Check if multiple predictions were found
    if [ ${{#founds[@]}} -gt 1 ]; then
        echo "Warning: found multiple predictions for ${{INSTANCE_ID}}"
    fi
    
    # Use the first found prediction
    local predsd="${{founds[0]}}"
    
    # Check if model_patch field exists
    if [ "$(echo "$predsd" | jq 'has("model_patch")')" != "true" ]; then
        echo "No model patch field found for ${{INSTANCE_ID}}, skipping ..."
        rm -rf "${{temp_dir}}"
        return 0
    fi
    
    # Check if model_patch field exists within model_patch
    if [ "$(echo "$predsd" | jq '.model_patch | has("model_patch")')" != "true" ]; then
        echo "No model patch field in model patch for ${{INSTANCE_ID}}, skipping ..."
        rm -rf "${{temp_dir}}"
        return 0
    fi
    
    # Check if model_patch is null
    if [ "$(echo "$predsd" | jq '.model_patch.model_patch == null')" = "true" ]; then
        echo "Model patch for ${{INSTANCE_ID}} is null, skipping ..."
        rm -rf "${{temp_dir}}"
        return 0
    fi
    
    # Ensure logs directory exists
    mkdir -p logs
    
    # Write to output file - extract the model_patch field and write as compact JSON on a single line
    echo "$predsd" | jq -c '.model_patch' > "logs/${{INSTANCE_ID}}_all_preds.jsonl"
    
    # Clean up temp directory
    rm -rf "${{temp_dir}}"
    
    # Running check
    {check_cmd}
}}
process_files"""


SWEAGENT_BF_CHECK_CMD = PATCH_CHECK_CMD.format(
   results_dir="sweb-sweagent-bf",
   check_cmd="""python codearena.py \
        --BugFixing \
        --predictions_path "logs/${INSTANCE_ID}_all_preds.jsonl" \
        --run_id sweagent_bf_check \
        --instance_ids "${INSTANCE_ID}" """,
)


SWEAGENT_TG_CHECK_CMD = PATCH_CHECK_CMD.format(
   results_dir="sweb-sweagent-tg",
   check_cmd="""python codearena.py \
        --TestGeneration \
        --predictions_path "logs/${INSTANCE_ID}_all_preds.jsonl" \
        --run_id sweagent_tg_check \
        --instance_ids "${INSTANCE_ID}" """,
)




SWEAGENT_BUGFIXING_CMD = """python baselines/sweagent/sweagent_regular.py \
-i data/codearena_instances.json \
-o logs \
-m gemini/gemini-2.5-flash-preview-04-17 \
-k $GEMINI_API_KEY \
--mode bugfixing \
--instance_ids $INSTANCE_ID"""



SWEAGENT_TESTGEN_CMD = """python baselines/sweagent/sweagent_regular.py \
-i data/codearena_instances.json \
-o logs \
-m gemini/gemini-2.5-flash-preview-04-17 \
-k $GEMINI_API_KEY \
--mode testgen \
--instance_ids $INSTANCE_ID"""



SWEAGENT_BUGFIXING_JAVA_CMD = """python baselines/sweagent/sweagent_regular.py \
-i data/codearena_instances_java.json \
-o logs \
-m gemini/gemini-2.5-flash-preview-04-17 \
-k $GEMINI_API_KEY \
--mode bugfixing_java \
--instance_ids $INSTANCE_ID"""


SWEAGENT_TESTGEN_JAVA_CMD = """python baselines/sweagent/sweagent_regular.py \
-i data/codearena_instances_java.json \
-o logs \
-m gemini/gemini-2.5-flash-preview-04-17 \
-k $GEMINI_API_KEY \
--mode testgen_java \
--instance_ids $INSTANCE_ID"""


COMMAND_MAP = {
    "sanity": SANITY_CMD,
    "bp-gen": BAD_PATCH_GEN_CMD,
    "agentless-check": AGENTLESS_CHECK_CMD,
    "sweagent-bf": SWEAGENT_BUGFIXING_CMD,
    "sweagent-bf-check": SWEAGENT_BF_CHECK_CMD,
    "sweagent-tg": SWEAGENT_TESTGEN_CMD,
    "sweagent-tg-check": SWEAGENT_TG_CHECK_CMD,
    "sweagent-bf-java": SWEAGENT_BUGFIXING_JAVA_CMD,
    "sweagent-tg-java": SWEAGENT_TESTGEN_JAVA_CMD,
}

def get_command(
    job_type: str
) -> str:
    if job_type not in COMMAND_MAP:
        raise RuntimeError(f"Invalid job type: {job_type}")
    
    return COMMAND_MAP[job_type]
