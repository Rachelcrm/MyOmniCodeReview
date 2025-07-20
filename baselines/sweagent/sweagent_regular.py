from pathlib import Path
import json
import logging
import tempfile
import base64
import subprocess
import os

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from jinja2 import Template
import pandas as pd

from sweagent.run.run import main as sweagent_main
from sweagent.run.run_single import RunSingle, RunSingleConfig
from sweagent.environment.swe_env import EnvironmentConfig, DockerDeploymentConfig
from sweagent.environment.repo import GithubRepoConfig
from sweagent.agent.agents import AgentConfig
from sweagent.agent.models import GenericAPIModelConfig
from sweagent.agent.problem_statement import TextProblemStatement, FileProblemStatement

CUR_DIR = Path(__file__).parent
DOTENV_PATH = CUR_DIR / '.env'

CONFIG_FILE_MAP = {
    "bugfixing": CUR_DIR / "bugfixing.yaml",
    "testgen": CUR_DIR / "testgen.yaml",
    "bugfixing-java": CUR_DIR / "bugfixing_java.yaml",
    "testgen-java": CUR_DIR / "testgen_java.yaml",
    "stylereview": CUR_DIR / "stylereview.yaml",
    "reviewfix": CUR_DIR / "reviewfix.yaml",
    "reviewfix-java": CUR_DIR / "reviewfix_java.yaml",
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# def run_sweagent_single(
#     instance: dict,
#     model_name: str,
#     output_dir: Path,
# ):

#     agent = AgentConfig(
#         model=GenericAPIModelConfig(
#             name=model_name,
#         ),
#     )

#     url = f"https://github.com/{instance['repo']}"

#     env = EnvironmentConfig(
#         deployment=DockerDeploymentConfig(image="python:3.12"),
#         repo=GithubRepoConfig(
#             github_url=url,
#             base_commit=instance['base_commit'],
#         ),
#         post_startup_commands=[],
#     )


#     # problem_statement = TextProblemStatement(
#     #     text=PROMPT_TEMPLATE.render(
#     #         issue=instance['problem_statement']
#     #     ),
#     #     id=instance['instance_id'],
#     # )

#     with tempfile.NamedTemporaryFile(delete_on_close=False, mode="w") as fp:
#         fp.write(
#             PROMPT_TEMPLATE.render(
#                 issue=instance['problem_statement']
#             )
#         )
#         fp.close()


#         problem_statement = FileProblemStatement(
#             path=Path(fp.name),
#             id=instance['instance_id'],
#         )

#         config = RunSingleConfig(
#             env=env,
#             agent=agent,
#             problem_statement=problem_statement,
#             output_dir=output_dir,
#             env_var_path=DOTENV_PATH,
#         )

#         RunSingle.from_config(config).run()


#     output_file_path = output_dir / problem_statement.id / (problem_statement.id + ".pred")
#     output = json.loads(output_file_path.read_text())

#     return None, output

class ArgumentTypeError(Exception):
    """An error from trying to convert a command line string to a type."""
    pass

def str2bool(v):
    """
    Minor helper function to convert string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")

def get_reviewfix_faux_problem_statement(instance: dict) -> str:
    if 'bad_patches' not in instance or len(instance['bad_patches']) == 0:
        logger.warning(f"Instance {instance['instance_id']} does not have any bad patches, cannot generate faux problem statement.")
        return None
    bad_patch = instance['bad_patches'][0]    
    # bad_patch = [bp for bp in instance['bad_patches'] if bp['source'] == 'badpatchllm'][0]
    problem_statement = instance['problem_statement']
    bad_patch_text = bad_patch['patch']
    review = bad_patch['review']

    faux_str = f"""Consider the following PR description:

      <pr_description>
      {problem_statement}
      </pr_description>

      Additionally, here is a previous patch attempt that failed to resolve this issue.

      <bad_patch>
      {bad_patch_text}
      </bad_patch>

      And here are is a review that attempts to explain why the patch failed:

      <review>
      {review}
      </review>

      Please carefully review the failed patch and its reviews. Use insight from them to **avoid repeating the same mistakes** and to **guide your reasoning** when implementing the fix."""
    return faux_str

def build_swe_agent_image(base_image_name, tag_name="swe-agent-image:latest"):
    """
    Builds a Docker image for SWE-agent with a dynamic base image.

    Args:
        base_image_name (str): The name of the existing base image (e.g., "ubuntu", "my-custom-image").
        tag_name (str): The name to give to the new SWE-agent image.
    """
    dockerfile_content = f"""
# Start with your existing base image
FROM {base_image_name}

# 1. Install Python 3, pip, and venv
RUN apt update && \\
    apt install -y python3 python3-pip python3-venv git && \\
    rm -rf /var/lib/apt/lists/*

# 2. Set Python 3 as the default 'python' command (optional but good practice)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 3. Create a virtual environment for SWE-agent's Python dependencies
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip setuptools

# 5. Clone the SWE-agent repository into a *sub-directory* to ensure
#    the root of the repo is clearly defined.
RUN $VIRTUAL_ENV/bin/pip install -e git+https://github.com/SWE-agent/SWE-agent@bb80cbe#egg=sweagent

# Set a default command or entrypoint if desired, though SWE-agent will likely override this
# CMD ["bash"]
"""

    # Create a temporary Dockerfile
    dockerfile_path = "Dockerfile.tmp"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    try:
        command = [
            "docker", "build",
            "-t", tag_name,
            "-f", dockerfile_path, 
            "." 
        ]

        print(f"Executing command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Docker build output:")
        print(process.stdout)
        if process.stderr:
            print("Docker build errors (if any):")
            print(process.stderr)
        print(f"Successfully built Docker image: {tag_name}")

    except subprocess.CalledProcessError as e:
        print(f"Error building Docker image: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    finally:
        # Clean up the temporary Dockerfile
        if os.path.exists(dockerfile_path):
            os.remove(dockerfile_path)


def run_sweagent_single(
    instance: dict,
    model_name: str,
    api_key: str | None,
    output_dir: Path,
    mode: str = "bugfixing",
    thinking_budget: int | None = None,
    use_apptainer: bool = False,
):

    url = f"https://github.com/{instance['repo']}"

    if mode not in CONFIG_FILE_MAP:
        raise RuntimeError(f"Unknown mode: {mode}")
    
    if 'java' in mode:
        base_image = f"mswebench/{instance['repo'].replace('/', '_m_')}:base"
        image = f"mswebench/{instance['repo'].replace('/', '_m_')}_with_python:base"
        build_swe_agent_image(base_image, tag_name=image)
    else:
        image = f"sca63/codearena:{instance['instance_id']}"

    print(f"Using image: {image}")
    config_file = CONFIG_FILE_MAP[mode]

    with tempfile.NamedTemporaryFile(delete_on_close=False, mode="w") as fp:

        if mode == 'reviewfix' or mode == 'reviewfix-java':
            # use the problem statement to inject prompt, hacky way to modify prompt easily
            prompt = get_reviewfix_faux_problem_statement(instance)
            if prompt is None:
                return "", ""
            fp.write(prompt)
        else:
            fp.write(instance['problem_statement'])

        fp.close()

        args = ["run"]

        if config_file is not None:
            args.extend([f"--config",  str(config_file)])

        args += [
            f"--agent.model.name={model_name}",
            f"--agent.model.per_instance_cost_limit=2.0",
            f"--env.repo.github_url={url}",
            f"--env.repo.base_commit={instance['base_commit']}",
            f"--env.deployment.image={image}",
            # override having /testbed be WORKDIR for docker image
            '--env.deployment.docker_args=["-w","/"]',
            f"--problem_statement.path={str(fp.name)}",
            f"--problem_statement.id={instance['instance_id']}",
            f"--output_dir={output_dir}",
        ]

        if api_key is not None:
            args.append(f"--agent.model.api_key={api_key}")

        if mode == 'stylereview':
            # apply gold patch upon starting env, so that agent can modify it based on pylint feedback
            commands = apply_patch_commands(instance["patch"], repo_name=instance["repo"].replace("/", "__"))

            args.append(
                f"--env.post_startup_commands={json.dumps(commands)}",     # note: !r gives Python‑style list
            )


        if thinking_budget is not None:
            if model_name.startswith("gemini"):
                args.append("""--agent.model.completion_kwargs={"thinking":{"type":"enabled","budget_tokens":""" + str(int(thinking_budget)) + """}}""")
            else:
                raise RuntimeError(f"Cannot use thinking budget with non-gemini model: {model_name}")
        
        args.append(f"--use_apptainer={str(use_apptainer).lower()}")

        sweagent_main(args)

    output_file_path = output_dir / instance['instance_id'] / (instance['instance_id']  + ".pred")
    output = json.loads(output_file_path.read_text())

    return None, output

def apply_patch_commands(patch: str, repo_name: str) -> list[str]:
    """
    Return a list of commands that apply the patch to the repo.

    1.  recreate /tmp/patch.diff inside the container
    2.  try git‑apply, fallback to patch -p1 --fuzz
    """
    b64 = base64.b64encode(patch.encode()).decode()
    return [
        # write file atomically
        f"echo '{b64}' | base64 -d > /tmp/patch.diff",
        # cd into repo and apply
        f"""cd /{repo_name} && (
                git apply --allow-empty -v /tmp/patch.diff ||
                patch --batch --fuzz=5 -p1 -i /tmp/patch.diff
            )""",
    ]

def main(
    input_tasks_path: Path,
    output_dir_path: Path,
    model_name: str,
    api_key: str | None,
    instance_ids: list[str] | None= None,
    mode: str = "bugfixing",
    thinking_budget: int | None = None,
    use_apptainer: bool = False,
):
    if input_tasks_path.exists():
        if input_tasks_path.suffix.endswith("json"):
            dataset = json.loads(input_tasks_path.read_text())
        elif input_tasks_path.suffix.endswith("jsonl"):
            dataset = [json.loads(i) for i in input_tasks_path.read_text().splitlines()]
        elif input_tasks_path.suffix.endswith("csv"):
            dataset = pd.read_csv(input_tasks_path).to_dict('records')
        else:
            raise RuntimeError(f"Data type ({input_tasks_path.suffix}) not supported")

    else:
        dataset = load_dataset(str(input_tasks_path))

    if isinstance(dataset, dict):
        dataset = dataset['test']

    if not (isinstance(dataset, list) and all(isinstance(d, dict) for d in dataset)):
        raise RuntimeError(f"Data folllows incorrect format")

    if instance_ids is not None:
        dataset = [d for d in dataset if d["instance_id"] in instance_ids]

    existing_ids = set()

    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir_path / "all_preds.jsonl"

    if output_file_path.exists():
        with open(output_file_path) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                existing_ids.add(instance_id)
    logger.info(f"Read {len(existing_ids)} already completed ids from {output_file_path}")

    basic_args = {
        "model_name_or_path": model_name,
    }

    with open(output_file_path, "a+") as f:
        for datum in tqdm(dataset, desc=f"Inference for {model_name}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue
            output_dict = {"instance_id": instance_id}
            output_dict.update(basic_args)
            full_output, model_patch = run_sweagent_single(datum, model_name=model_name, output_dir=output_dir_path, api_key=api_key, mode=mode, thinking_budget=thinking_budget, use_apptainer=use_apptainer)
            output_dict["full_output"] = full_output
            output_dict["model_patch"] = model_patch
            print(json.dumps(output_dict), file=f, flush=True)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_tasks", type=str, required=True)
    parser.add_argument("--instance_ids", type=str, required=False, default=None)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="gemini/gemini-2.5-flash")
    parser.add_argument("-k", "--api_key", type=str, default=None)
    parser.add_argument("--mode", type=str, default="bugfixing", choices=["bugfixing", "testgen", "bugfixing-java", "testgen-java", "stylereview", "reviewfix", "reviewfix-java"])
    parser.add_argument("--thinking_budget", type=int, default=0)
    parser.add_argument("--use_apptainer", type=str2bool, default=False, help="run with docker or apptainer")
    args = parser.parse_args()

    main(
        input_tasks_path=Path(args.input_tasks),
        output_dir_path=Path(args.output_dir),
        model_name=args.model_name,
        instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
        api_key=args.api_key,
        mode=args.mode,
        # thinking_budget=args.thinking_budget,
        use_apptainer=args.use_apptainer,
    )

