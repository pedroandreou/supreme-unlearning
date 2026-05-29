import os
import json
import time
import signal
import sys
import threading
import psutil
from src.eval_metrics.resource_consumption import (
    get_visible_gpu_ids,
)


class ProcessTracker:
    """
    Tracks and manages Python processes with activity monitoring.

    Provides functionality to:
    - Track main and child processes
    - Save process information to temporary files
    - Monitor log files for activity, detect stalls and auto-terminate processes on inactivity or verify completion (180s timeout)
    - Clean up processes on user interruption (CTRL+C/SIGINT) and system/normal termination (SIGTERM)
    - Kill specific processes based on script type, model, or dataset given by the user

    Used for preventing zombie processes and stalled executions when scripts
    terminate unexpectedly or need to be stopped manually.
    """

    # Map script types to their completion messages
    SCRIPT_CONFIGS = {
        "train": {
            "completion_msg": "=== Training completed",
        },
        "unlearn": {
            "completion_msg": "=== Unlearning completed",
        },
    }

    def __init__(self, fabric, script_type, model_name, dataset_name, **kwargs):
        """
        Initialize process tracker with script identification and metadata.

        Args:
            script_type (str): Type of script ('train' or 'unlearn')
            model_name (str): Name of the model being used
            dataset_name (str): Name of the dataset
            **kwargs: Additional tracking metadata
        """

        self.fabric = fabric
        if script_type not in self.SCRIPT_CONFIGS:
            raise ValueError(
                f"Invalid script_type: {script_type}. Must be one of {list(self.SCRIPT_CONFIGS.keys())}"
            )

        self.script_type = script_type
        self.tracking_dir = "/tmp/ml_processes"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._cleanup_in_progress = False
        self._stop_monitoring = False

        # Get script-specific configuration
        script_config = self.SCRIPT_CONFIGS[script_type]
        self.completion_msg = script_config["completion_msg"]
        self.timeout_seconds = 180  # Time without activity before declaring bottleneck
        self.check_interval = 10  # How often to check for activity
        self.last_check_time = time.time()

        # Find the actual log file being used by the process
        self.log_file = self._find_log_file()
        fabric.print(f"\nDetected log file: {self.log_file}")

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_log_activity)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # Create unique identifier - include unlearning strategy in ID if present
        if script_type == "unlearn" and "type_of_unlearning_strategy" in kwargs:
            self.unique_id = f"{script_type}_{model_name}_{dataset_name}_{kwargs['type_of_unlearning_strategy']}_{timestamp}"
        else:
            self.unique_id = f"{script_type}_{model_name}_{dataset_name}_{timestamp}"

        # Base process info
        self.process_info = {
            "main_pid": os.getpid(),
            "script_type": script_type,
            "model": model_name,
            "dataset": dataset_name,
            "gpu_ids": get_visible_gpu_ids(),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "child_pids": [],
            "log_file": self.log_file,  # Store the detected log file
        }

        # Add additional info from kwargs
        self.process_info.update(kwargs)

        os.makedirs(self.tracking_dir, exist_ok=True)

        # Set up file path
        self.info_file = os.path.join(self.tracking_dir, f"{self.unique_id}.json")

        # Save initial info
        self._save_info()

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _find_log_file(self):
        """
        Find the actual log file being used by the process.
        First checks LOG_DIR for unlearning experiments, then falls back to output_log_files.
        """
        try:
            # First check if we're in an unlearning experiment
            log_dir = os.environ.get("LOG_DIR")
            if log_dir:
                # For unlearning experiments, monitor the nohup.out in the LOG_DIR
                nohup_path = os.path.join(log_dir, "nohup.out")
                if os.path.exists(nohup_path):
                    return nohup_path

            # For training/other experiments, check output_log_files directory
            output_logs_dir = os.path.join("logs", "output_log_files")

            process = psutil.Process(os.getpid())
            for fd in process.open_files():
                # Only consider files in our logs directories
                if output_logs_dir in fd.path or (log_dir and log_dir in fd.path):
                    if (
                        fd.path.endswith(".log")
                        or os.path.basename(fd.path) == "nohup.out"
                    ):
                        return fd.path

            # If no log file found in current process, check parent process
            parent = process.parent()
            if parent:
                for fd in parent.open_files():
                    if output_logs_dir in fd.path or (log_dir and log_dir in fd.path):
                        if (
                            fd.path.endswith(".log")
                            or os.path.basename(fd.path) == "nohup.out"
                        ):
                            return fd.path

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

        # If no log file found, return None
        return None

    def _save_info(self):
        """Save process information to file"""
        with open(self.info_file, "w") as f:
            json.dump(self.process_info, f)

    def update_info(self, **kwargs):
        """Update process information"""
        self.process_info.update(kwargs)
        self._save_info()

    def add_child_pid(self, pid):
        """Add a child process ID"""
        self.process_info["child_pids"].append(pid)
        self._save_info()

    def _monitor_log_activity(self):
        """
        Monitor log file for activity and detect potential bottlenecks.

        Sleeps for the full timeout period (default 3 minutes) before checking
        for activity. If no new activity is detected and no completion message
        is found, terminates the processes.
        """
        if not self.log_file:
            self.fabric.print(
                "\nNo log file detected for monitoring. Process tracking will be limited."
            )
            return

        self.fabric.print(f"\nStarting log monitoring for file: {self.log_file}")
        while not self._stop_monitoring:
            try:
                if os.path.exists(self.log_file):
                    initial_modified = os.path.getmtime(self.log_file)

                    # Sleep for the full timeout period
                    time.sleep(self.timeout_seconds)

                    if self._stop_monitoring:  # Check if we should exit
                        break

                    current_modified = os.path.getmtime(self.log_file)

                    # If file hasn't been modified during our sleep
                    if current_modified == initial_modified:
                        # Check if completion message is present
                        with open(self.log_file, "r") as f:
                            content = f.read()
                            if self.completion_msg not in content:
                                self.fabric.print(
                                    f"\nBottleneck detected - no activity for {self.timeout_seconds} seconds"
                                )
                                self.fabric.print(f"Script type: {self.script_type}")
                                self.fabric.print(f"Log file: {self.log_file}")
                                self.fabric.print(
                                    f"Last modified: {time.ctime(current_modified)}"
                                )
                                self.cleanup()
                                break
                            else:
                                self.fabric.print(
                                    "\nCompletion message found in log file"
                                )
                                break

                else:
                    # Only print if the file doesn't exist on first check
                    if not hasattr(self, "_file_existence_checked"):
                        self.fabric.print(f"\nLog file does not exist: {self.log_file}")
                        self._file_existence_checked = True

                time.sleep(1)  # Brief sleep before checking file existence again
            except Exception as e:
                self.fabric.print(f"Error monitoring log file: {e}")
                time.sleep(1)

    def cleanup(self):
        """Clean up process tracking files and kill child processes"""
        if self._cleanup_in_progress:  # Prevent recursive cleanup
            return

        try:
            self._cleanup_in_progress = True
            self._stop_monitoring = True  # Stop the monitoring thread

            def force_kill_process(pid, is_main=False):
                try:
                    proc = psutil.Process(pid)
                    # Get all child processes before killing the parent
                    children = proc.children(recursive=True)

                    # First try SIGTERM
                    proc.terminate()

                    # Wait for up to 3 seconds for graceful termination
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        # If SIGTERM didn't work, use SIGKILL
                        self.fabric.print(
                            f"Process {pid} didn't terminate gracefully, forcing kill..."
                        )
                        proc.kill()

                    # Kill all child processes that might still be running
                    for child in children:
                        try:
                            if child.is_running():
                                child.kill()
                        except psutil.NoSuchProcess:
                            pass

                    msg = "main" if is_main else "child"
                    self.fabric.print(
                        f"Successfully terminated {msg} process {pid} and its children"
                    )

                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    self.fabric.print(f"Error killing process {pid}: {e}")

            # Kill child processes first
            for child_pid in self.process_info.get("child_pids", []):
                force_kill_process(child_pid)

            # Handle the main process
            try:
                main_pid = self.process_info["main_pid"]
                current_pid = os.getpid()

                if current_pid != main_pid:  # Kill other processes immediately
                    force_kill_process(main_pid, is_main=True)
                else:  # If we are the main process
                    # Kill all child processes of the current process
                    try:
                        current_proc = psutil.Process(current_pid)
                        children = current_proc.children(recursive=True)

                        self.fabric.print(
                            f"Found {len(children)} child processes to terminate"
                        )

                        # Terminate child processes first
                        for child in children:
                            try:
                                self.fabric.print(
                                    f"Terminating child process {child.pid}"
                                )
                                child.terminate()
                            except psutil.NoSuchProcess:
                                pass

                        # Wait a bit for graceful termination
                        time.sleep(2)

                        # Force kill any remaining child processes
                        for child in children:
                            try:
                                if child.is_running():
                                    self.fabric.print(
                                        f"Force killing child process {child.pid}"
                                    )
                                    child.kill()
                            except psutil.NoSuchProcess:
                                pass

                        self.fabric.print(
                            "Main process cleanup complete - terminating children and self"
                        )

                    except psutil.NoSuchProcess:
                        pass

                    # Force exit the main process after cleaning up children
                    os._exit(
                        1
                    )  # Use os._exit instead of sys.exit for more forceful termination

            except ProcessLookupError:
                pass

            # Clean up the tracking file
            try:
                if os.path.exists(self.info_file):
                    os.remove(self.info_file)
            except OSError:
                pass

        finally:
            self._cleanup_in_progress = False

    def _signal_handler(self, signum, frame):
        """Handle termination signals by cleaning up processes"""
        self._stop_monitoring = True  # Stop monitoring before cleanup
        self.cleanup()
        sys.exit(0)  # Exit after cleanup

    @staticmethod
    def kill_processes(script_type=None, model=None, dataset=None):
        """
        Kill processes matching the specified criteria
        Example: ProcessTracker.kill_processes(script_type='train', model='ResNet18', dataset='Cifar10')
        """
        tracking_dir = "/tmp/ml_processes"
        if not os.path.exists(tracking_dir):
            return

        for filename in os.listdir(tracking_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(tracking_dir, filename)
            try:
                with open(filepath, "r") as f:
                    info = json.load(f)

                should_kill = True
                if script_type and info.get("script_type") != script_type:
                    should_kill = False
                if model and info.get("model") != model:
                    should_kill = False
                if dataset and info.get("dataset") != dataset:
                    should_kill = False

                if should_kill:
                    pid = info.get("main_pid")
                    if pid:
                        try:
                            os.kill(pid, signal.SIGTERM)
                            # Kill child processes if any
                            for child_pid in info.get("child_pids", []):
                                try:
                                    os.kill(child_pid, signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                        except ProcessLookupError:
                            pass
                    try:
                        os.remove(filepath)
                    except OSError:
                        continue
            except (
                Exception
            ):  # Using Exception to catch most errors but not system exits/interrupts
                continue
