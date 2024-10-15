import os
import sys
import subprocess


def main():
    # Get the GPU ID and the additional arguments from the command line
    if len(sys.argv) < 2:
        print("Usage: python train.py <GPU_ID> [additional arguments]")
        sys.exit(1)

    gpus = sys.argv[1]
    py_args = sys.argv[2:]

    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Prepare the command to run
    command = ["python", "main.py"] + py_args

    # Print the command being run (similar to 'set -x' in bash)
    print(f"Running command: {' '.join(command)}")

    # Run the command
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()