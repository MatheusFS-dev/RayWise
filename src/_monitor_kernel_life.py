# ---------------------------------------------------------------------------- #
#                      Authored by Matheus Ferreira Silva                      #
#                           github.com/MatheusFS-dev                           #
# ---------------------------------------------------------------------------- #

"""
Kernel Process Monitor
----------------------
This script reads a PID and monitors whether the corresponding
process is still running. If the process dies, an alert is logged and an
email notification is sent. You can optionally supply a custom title
to identify the monitored process.
"""

import time
import psutil
import argparse
from typing import Optional


def send_alert_email(pid: int, status: str, custom_title: Optional[str] = None) -> None:
    """
    Sends an email notification about the process status.

    Args:
        pid (int): The process ID of the monitored process.
        status (str): Either "crashed" or "terminated".
        custom_title (Optional[str]): Custom name for the monitored process;
            if None, defaults to "Kernel process".
    """
    # Determine the display title
    process_title = custom_title or "Kernel process"

    try:
        from utils.email_api import send_email
    except ImportError as imp_err:
        print(f"[ERROR] Email API import failed: {imp_err}")
        return

    email_subject = f"{process_title} (PID {pid}) {status.capitalize()}"
    email_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;
                     background-color: #f9f9f9; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background: #fff; padding: 20px;
                    border: 1px solid #ddd; border-radius: 8px;">
            <h2 style="color: #d9534f; text-align: center;">
                ⚠️ {process_title} Alert: {status.capitalize()}
            </h2>
            <p style="font-size: 16px; color: #444;">
                <strong>Dear User,</strong>
            </p>
            <p style="font-size: 18px; color: #333;">
                The process "<strong>{process_title}</strong>" with PID
                <strong style="color: #d9534f;">{pid}</strong> has {status}.
            </p>
            <p style="font-size: 16px; color: #555;">
                Please check the system logs and take necessary actions to resolve
                the issue.
            </p>
            <p style="text-align: center; font-size: 16px;">
                <strong style="color: #d9534f;">⚠️ Status:</strong>
                <span style="color: #d9534f;">{status.capitalize()}</span>
            </p>
            <footer style="margin-top: 20px; text-align: center; font-size: 14px;
                           color: #888;">
                <p>Best regards,</p>
                <p><strong>The Monitoring Team</strong></p>
            </footer>
        </div>
        </body>
    </html>
    """

    try:
        send_email(
            subject=email_subject,
            body=email_body,
            recipients_file="./json/recipients.json",
            credentials_file="./json/credentials.json",
            text_type="html",
        )
        print(f"[INFO] Email sent: {process_title} (PID {pid}) {status}.")
    except Exception as e:
        print(f"[ERROR] Failed to send email for {process_title} (PID {pid}): {e}")


def monitor_process(
    pid: int,
    check_interval: int = 10,
    custom_title: Optional[str] = None
) -> None:
    """
    Monitors a running process by checking if the PID exists. When the
    process stops, logs an alert and sends an email.

    Args:
        pid (int): The process ID to monitor.
        check_interval (int): Seconds between health checks.
        custom_title (Optional[str]): Custom name for the monitored process.
    """
    process_title = custom_title or "Kernel process"
    print(f"[INFO] Monitoring '{process_title}' with PID {pid}...")

    while True:
        try:
            proc = psutil.Process(pid)
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                print(f"[ALERT] {process_title} (PID {pid}) has crashed!")
                send_alert_email(pid, "crashed", custom_title)
                break

        except psutil.NoSuchProcess:
            print(f"[ALERT] {process_title} (PID {pid}) no longer exists!")
            send_alert_email(pid, "terminated", custom_title)
            break

        time.sleep(check_interval)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Monitor a process by PID and send alerts if it stops."
    )
    parser.add_argument(
        "--pid", type=int, required=True,
        help="Process ID to monitor."
    )
    parser.add_argument(
        "--interval", type=int, default=10,
        help="Check interval in seconds (default: 10)."
    )
    parser.add_argument(
        "--custom-title", type=str, default=None,
        help="Optional custom name for the monitored process."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    monitor_process(
        pid=args.pid,
        check_interval=args.interval,
        custom_title=args.custom_title
    )
