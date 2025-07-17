"""
This script auto launches the main application and restarts it if it crashes.

Command line usage:
    python _auto_run.py nas_cnn1d_flat_v9.ipynb --title "ML Training"

    For Anaconda environments, use:
    /home/matheus/anaconda3/envs/tf-optuna/bin/python _auto_run.py nas_cnn1d_flat_v9.ipynb --title "ML Training"
"""

# —————————————————————————————————— Imports ————————————————————————————————— #
import sys
import argparse
from araras.runtime.monitoring import run_auto_restart

# ————————————————————————————————— Constants ———————————————————————————————— #
YELLOW = "\033[93m"
RESET = "\033[0m"
ORANGE = "\033[38;5;208m"

# ———————————————————— Warnings and Executable Information ——————————————————— #
print(f"{YELLOW}=" * 100 + f"{RESET}")
print(" " * 42 + f"{YELLOW}AUTO RUN SCRIPT{RESET}\n")
print(f"{YELLOW}[WARNING] Run this script with sudo if the other processes require it!{RESET}")
print(
    f"{YELLOW}[WARNING] If using Anaconda Python environment, ensure you are using the correct interpreter.{RESET}"
)
print(f"{ORANGE}    >>> {sys.executable} is running this script{RESET}")
print(f"{YELLOW}=" * 100 + f"\n\n{RESET}")
# ———————————————————————————————————————————————————————————————————————————— #


def main():
    parser = argparse.ArgumentParser(description="Auto restart application launcher")
    parser.add_argument("file_path", type=str, help="Path to the file to run")
    parser.add_argument("--success-flag-file", default="/tmp/success.flag", help="Path to success flag file")
    parser.add_argument("--title", default=None, help="Title for the application")
    parser.add_argument("--max-restarts", type=int, default=10, help="Maximum number of restarts")
    parser.add_argument("--restart-delay", type=float, default=5.0, help="Delay between restarts in seconds")
    parser.add_argument(
        "--restart-after-delay", type=float, default=0.0, help="Restart run after delay in seconds"
    )

    args = parser.parse_args()

    run_auto_restart(
        file_path=args.file_path,
        success_flag_file=args.success_flag_file,
        title=args.title if args.title else args.file_path,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay,
        restart_after_delay=args.restart_after_delay,
        supress_tf_warnings=True,
        resource_usage_log_file=f"{args.title}.log" if args.title else f"{args.file_path}.log"
    )


if __name__ == "__main__":
    main()
