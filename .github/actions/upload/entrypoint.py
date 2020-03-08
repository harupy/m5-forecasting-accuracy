import os
import json
import tempfile
import subprocess
import shutil


def to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def run_command(command, verbose=True):
    print("Executing:", command)
    p = subprocess.Popen(
        [command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = p.communicate()

    if verbose:
        print("----- stdout -----\n")
        print(stdout.decode("utf-8"))

        print("----- stderr -----\n")
        print(stderr.decode("utf-8"))

    return p.returncode


def create_kernel_meta(id, title, code_file, competition_sources):
    return {
        "id": id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "false",
        "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": [],
        "competition_sources": competition_sources,
        "kernel_sources": [],
    }


def get_action_input(name, as_list=False):
    action_input = os.getenv(f"INPUT_{name.upper()}")
    if as_list:
        # Ignore empty and comment lines.
        return [
            x
            for x in action_input.split("\n")
            if x.strip() != "" and not x.startswith("#")
        ]

    return action_input


def main():
    id = get_action_input("id")
    title = get_action_input("title")
    code_file = get_action_input("code_file")
    competition_sources = get_action_input("competition_sources", as_list=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save kernel metadata to tmpdir.
        meta = create_kernel_meta(
            id, title, os.path.basename(code_file), competition_sources
        )
        to_json(meta, os.path.join(tmpdir, "kernel-metadata.json"))

        # Copy script to tmpdir.
        dst = os.path.join(tmpdir, os.path.basename(code_file))
        shutil.copyfile(code_file, dst)

        run_command(f"kaggle kernels push -p {tmpdir}")
        run_command(f"kaggle kernels status")


if __name__ == "__main__":
    main()
