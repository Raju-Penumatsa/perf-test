# resources/log_collector.py

import os
import pandas as pd
from datetime import datetime


class LogCollector:
    """
    LogCollector handles writing system performance and test metrics
    into CSV files for later analysis and dashboard visualization.
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize LogCollector with log directory path.
        Creates the directory if it doesn't exist.
        """
        self.log_dir = os.path.join(os.path.dirname(__file__), '..', log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

    def collect(self, data: dict, filename: str = None):
        """
        Collect and append a performance run's data to a CSV log file.

        Args:
            data (dict): Dictionary of metric names and values.
            filename (str, optional): Specific filename for the log.
                If None, uses timestamp-based default.

        Returns:
            str: Path to the CSV log file written.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"run_{timestamp}.csv"

        file_path = os.path.join(self.log_dir, filename)

        df = pd.DataFrame([data])

        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, mode='w', header=True, index=False)

        return file_path
