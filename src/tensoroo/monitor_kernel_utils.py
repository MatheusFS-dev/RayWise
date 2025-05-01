"""
This module provides utilities for launching and managing kernel monitoring processes
in a variety of terminal emulators. It includes functions to spawn external processes
that monitor the current kernel or training job, ensuring isolation and ease of use.

Functions:
    - launch_process: Launches a monitoring script in a new terminal emulator window.
    - launch_kernel_monitor: A safe wrapper for launching the kernel monitor with fallback.

Example Usage:
    launch_kernel_monitor(resume_training_path="my_training_job", script_path="path/to/_monitor_kernel_life.py")
"""


import os
import shutil
import subprocess
from typing import List, Optional
from IPython.display import display, HTML


def launch_process(
    custom_title: str, script_path: Optional[str] = None
) -> Optional[subprocess.Popen]:
    """
    Launches a new terminal emulator window that runs a kernel monitoring script.

    This function is typically used to spawn an external process that monitors
    the current kernel or training job. It tries multiple terminal emulators (XFCE, GNOME, xterm, Konsole),
    and runs the monitor script in a new session to ensure it is isolated from the main job.

    Logic:
        -> Get PID of current process
        -> Build the shell command to invoke the monitor
        -> Try launching it in a variety of terminal emulators
        -> Return the subprocess object if successful

    Args:
        custom_title (str): Identifier or custom label passed to the monitor script.
        script_path (Optional[str]): Absolute or relative path to `_monitor_kernel_life.py`.
                                     If None, defaults to a file in the current directory.

    Returns:
        Optional[subprocess.Popen]: Process object for the launched monitor script, or None if failed.

    Raises:
        RuntimeError: If no supported terminal emulator is found on the system.
    """
    # Resolve path to the monitoring script
    monitor_script = (
        os.path.abspath(script_path) if script_path else os.path.abspath("_monitor_kernel_life.py")
    )

    # Get current process PID to monitor
    pid = os.getpid()

    # Construct the command to execute in the terminal
    cmd = f'python3 "{monitor_script}" ' f"--pid {pid} --custom-title {custom_title}; exec bash"

    # Ordered list of terminal launch commands to try
    terminals: List[List[str]] = [
        ["xfce4-terminal", "--disable-server", "--hold", "-e", f'bash -c "{cmd}"'],
        ["gnome-terminal", "--disable-factory", "--", "bash", "-i", "-c", cmd],
        ["xterm", "-hold", "-e", cmd],
        ["konsole", "--hold", "-e", f'bash -c "{cmd}"'],
    ]

    # Iterate through terminal options and launch first successful one
    for term_cmd in terminals:
        emulator = term_cmd[0]
        if shutil.which(emulator):  # Check if emulator is installed
            try:
                proc = subprocess.Popen(term_cmd, preexec_fn=os.setpgrp)  # Run in new process group
                print(f"[INFO] Launched monitor in {emulator} (PID={pid})")
                return proc
            except Exception as launch_err:
                print(f"[WARN] Failed to launch {emulator}: {launch_err}")
                continue

    # All terminal attempts failed — raise an error
    raise RuntimeError(
        "No supported terminal emulator found; " "install gnome-terminal, xfce4-terminal, konsole, or xterm."
    )


def launch_kernel_monitor(custom_title: str, script_path: Optional[str] = None) -> None:
    """
    A safe wrapper for launching the kernel monitor that handles failures gracefully.

    If no terminal emulator can be launched or the script fails to run,
    this function will show a fallback HTML message suggesting manual launch.

    Args:
        custom_title (str): Identifier or custom title passed to the monitoring script.
        script_path (Optional[str]): Path to the monitoring script file.
    
    Returns:
        Optional[subprocess.Popen]: Process object for the launched monitor script, or None if failed.
    """
    try:
        proc = launch_process(custom_title, script_path)
        return proc
    except Exception as e:
        pid = os.getpid()
        print(f"[ERROR] Auto launching kernel monitoring failed: {e}\n")
        display(
            HTML(
                f"Call the monitor script manually: "
                f'<span style="color: orange;">'
                f"python _monitor_kernel_life.py --pid {pid}"
                f"</span>"
            )
        )
