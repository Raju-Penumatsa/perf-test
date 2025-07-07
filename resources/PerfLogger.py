import platform
import psutil
import wmi
import os
import time
from robot.api.deco import keyword
from log_collector import LogCollector


class PerfLogger:
    def __init__(self):
        self.collector = LogCollector()
        self.w = wmi.WMI()

    @keyword
    def collect_system_metadata(self):
        """Collects system metadata like Windows version, RAID controller, hostname, etc."""
        try:
            raid_controllers = self.w.Win32_SCSIController()
            raid_name = raid_controllers[0].Name if raid_controllers else "Not Found"
        except Exception:
            raid_name = "Error retrieving RAID"

        return {
            "hostname": platform.node(),
            "os": platform.platform(),
            "processor": platform.processor(),
            "raid_controller": raid_name
        }

    @keyword
    def collect_performance_metrics(self):
        """Collect CPU, memory, disk, network, fan speeds, temps, etc."""
        cpu_percent = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()

        # Add other psutil metrics as needed

        return {
            "cpu_percent": cpu_percent,
            "memory_used_MB": mem.used / (1024*1024),
            "memory_percent": mem.percent,
            "disk_used_GB": disk.used / (1024*1024*1024),
            "disk_percent": disk.percent,
            "net_bytes_sent": net_io.bytes_sent,
            "net_bytes_recv": net_io.bytes_recv
        }

    @keyword
    def log_performance_result(self, extra_data=None):
        """Collects system metadata and performance metrics, merges with extra_data and writes to log."""
        data = {}
        data.update(self.collect_system_metadata())
        data.update(self.collect_performance_metrics())
        if extra_data:
            data.update(extra_data)
        filepath = self.collector.collect(data)
        return filepath

    @keyword
    def run_storage_test(self, filesize_mb):
        """
        Runs a storage test by writing and reading a file of given size (MB).
        Returns dictionary with 'read_time_sec' and 'write_time_sec'.
        """
        filename = "testfile.tmp"
        size_bytes = int(filesize_mb) * 1024 * 1024

        # Create dummy data to write
        data = b"0" * size_bytes

        # Write test
        start_write = time.time()
        with open(filename, "wb") as f:
            f.write(data)
        write_time = time.time() - start_write

        # Read test
        start_read = time.time()
        with open(filename, "rb") as f:
            _ = f.read()
        read_time = time.time() - start_read

        # Clean up file
        os.remove(filename)

        return {"read_time_sec": read_time, "write_time_sec": write_time, "filesize": filesize_mb}