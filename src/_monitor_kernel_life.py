# ---------------------------------------------------------------------------- #
#                      Authored by Matheus Ferreira Silva                      #
#                           github.com/MatheusFS-dev                           #
# ---------------------------------------------------------------------------- #

"""
Kernel Process Monitor
----------------------
This script reads a PID from a file and monitors whether the corresponding 
process is still running. If the process dies, an alert is logged and an 
email notification is sent.

Usage:
    python monitor.py --pid_path kernel_pid.txt

Example:
    Saving the kernel PID to a file:
    ```python
    import os
    pid = os.getpid()
    with open("kernel_pid.txt", "w") as f:
        f.write(str(pid))
    print(f"[INFO] Kernel PID saved to kernel_pid.txt: {pid}")
    ```
"""

import os
import time
import psutil
import argparse


def read_pid_from_file(pid_file: str) -> int | None:
    """
    Reads the kernel PID from the given file.

    Args:
        pid_file (str): Path to the file storing the PID.

    Returns:
        int | None: The PID if found and valid, otherwise None.
    """
    if not os.path.exists(pid_file):
        print(f"[ERROR] PID file '{pid_file}' not found!")
        return None

    try:
        with open(pid_file, "r") as f:
            return int(f.read().strip())
    except (ValueError, FileNotFoundError):
        print(f"[ERROR] Invalid PID file format: '{pid_file}'")
        return None


def send_alert_email(pid: int, status: str):
    """
    Sends an email notification about the kernel process status.

    Args:
        pid (int): The process ID of the monitored kernel.
        status (str): Either "crashed" or "terminated".
    """
    try:
        from utils.email_api import send_email

        # Subject and message based on the status
        email_subject = f"Kernel Process {pid} {status.capitalize()}"
        email_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f9f9f9; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: #fff; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                <h2 style="color: #d9534f; text-align: center;">⚠️ Kernel Alert: {status.capitalize()}</h2>
                <p style="font-size: 16px; color: #444;">
                <strong>Dear User,</strong>
                </p>
                <p style="font-size: 18px; color: #333;">
                The kernel process with PID <strong style="color: #d9534f;">{pid}</strong> has {status}.
                </p>
                <p style="font-size: 16px; color: #555;">
                Please check the system logs and take necessary actions to resolve the issue.
                </p>
                <p style="text-align: center; font-size: 16px;">
                <strong style="color: #d9534f;">⚠️ Status:</strong> <span style="color: #d9534f;">{status.capitalize()}</span>
                </p>
                <footer style="margin-top: 20px; text-align: center; font-size: 14px; color: #888;">
                <p>Best regards,</p>
                <p><strong>The Monitoring Team</strong></p>
                </footer>
            </div>
            </body>
        </html>
        """
        send_email(
            subject=email_subject,
            body=email_body,
            recipients_file="./json/recipients.json",
            credentials_file="./json/credentials.json",
            text_type="html",
        )
        print(f"[INFO] Email sent: Kernel process {pid} {status}.")
    except Exception as e:
        print(f"[ERROR] Failed to send email notification for PID {pid}: {e}")


def monitor_kernel(pid: int, check_interval: int = 3):
    """
    Monitors a running kernel process by checking if the PID exists.

    Args:
        pid (int): The process ID of the kernel to monitor.
        check_interval (int): Time in seconds between health checks.
    """
    print(f"[INFO] Monitoring kernel with PID {pid}...")

    while True:
        try:
            process = psutil.Process(pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                alert_message = f"[ALERT] Kernel process {pid} has crashed!"
                print(alert_message)

                # Send crash email
                send_alert_email(pid, "crashed")
                break
        except psutil.NoSuchProcess:
            alert_message = f"[ALERT] Kernel process {pid} no longer exists!"
            print(alert_message)

            # Send termination email
            send_alert_email(pid, "terminated")
            break

        time.sleep(check_interval)


def start_monitoring(pid_path: str, check_interval: int = 3):
    """
    Starts the kernel monitoring process by reading the PID from a file.

    Args:
        pid_path (str): Path to the file containing the PID.
        check_interval (int): Time in seconds between health checks.
    """
    pid = read_pid_from_file(pid_path)
    if pid is None:
        print("[ERROR] No valid PID found. Exiting.")
        return

    monitor_kernel(pid, check_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor a kernel process by PID.")
    parser.add_argument("--pid_path", type=str, required=True, help="Path to the file containing the PID")
    parser.add_argument("--interval", type=int, default=3, help="Check interval in seconds (default: 3)")

    args = parser.parse_args()
    start_monitoring(args.pid_path, args.interval)
