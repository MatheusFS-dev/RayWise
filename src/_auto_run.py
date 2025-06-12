"""
This script auto launches the main application and restarts it if it crashes.

Usage example:
    run_auto_restart(
        file_path="./nas_cnn1d_flat_v9.ipynb",
        success_flag_file="/tmp/success.flag",
        title="ML Training",
        max_restarts=10,
        restart_delay=5.0,
    )

Command line usage:
    python _auto_run.py nas_cnn1d_flat_v9.ipynb --title "ML Training"
"""

import argparse
from araras.kernel.restarting_monitoring import run_auto_restart


def main():
    parser = argparse.ArgumentParser(description="Auto restart application launcher")
    parser.add_argument("file_path", type=str, help="Path to the file to run")
    parser.add_argument("--success-flag-file", default="/tmp/success.flag", help="Path to success flag file")
    parser.add_argument("--title", default=None, help="Title for the application")
    parser.add_argument("--max-restarts", type=int, default=3, help="Maximum number of restarts")
    parser.add_argument("--restart-delay", type=float, default=10.0, help="Delay between restarts in seconds")

    args = parser.parse_args()

    run_auto_restart(
        file_path=args.file_path,
        success_flag_file=args.success_flag_file,
        title=args.title if args.title else args.file_path,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay,
    )


if __name__ == "__main__":
    main()
