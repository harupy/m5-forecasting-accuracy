import os
import json
import tempfile
import subprocess
import shutil


def to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def run_command(command):
    print("Executing:", command)
    p = subprocess.Popen(
        [command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = p.communicate()
    return p.returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


def create_kernel_meta(id, title, code_file, competition_source):
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
        "competition_sources": [competition_source],
        "kernel_sources": [],
    }


def get_action_input(name):
    return os.getenv(f"INPUT_{name.upper()}")


def main():
    id = get_action_input("id")
    title = get_action_input("title")
    code_file = get_action_input("code_file")
    competition_source = get_action_input("competition_source")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save kernel metadata to tmpdir.
        meta = create_kernel_meta(id, title, code_file, competition_source)
        to_json(meta, os.path.join(tmpdir, "kernel-metadata.json"))

        # Copy script to tmpdir.
        dst = os.path.join(tmpdir, os.path.basename(code_file))
        shutil.copyfile(code_file, dst)

        run_command(f"kaggle kernels push -p {tmpdir}")
        run_command(f"kaggle kernels status")


if __name__ == "__main__":
    main()
