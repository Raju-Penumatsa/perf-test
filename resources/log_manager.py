# resources/log_manager.py

import os
import glob
import pandas as pd


class LogManager:
    """
    LogManager handles locating, reading, and filtering log files
    from the specified logs directory.
    """

    def __init__(self, log_dir: str):
        """
        Initialize LogManager with the path to the log directory.
        """
        self.log_dir = log_dir

    def get_log_files_df(self) -> pd.DataFrame:
        """
        Scan the log directory and return a summary DataFrame of available CSV logs.
        """
        files = glob.glob(os.path.join(self.log_dir, "*.csv"))
        data = []
        for f in files:
            try:
                df = pd.read_csv(f, nrows=1)
                filesize = df.get("filesize", [None])[0]
                data.append({"filename": os.path.basename(f), "path": f, "filesize_MB": filesize})
            except Exception:
                continue
        return pd.DataFrame(data)

    def read_latest_logs(self, filesizes=None, max_runs=3) -> pd.DataFrame:
        """
        Read the most recent log files filtered by filesize and return combined DataFrame.
        """
        files = glob.glob(os.path.join(self.log_dir, "*.csv"))
        file_infos = []
        for f in files:
            try:
                df = pd.read_csv(f, nrows=1)
                filesize = df.get("filesize", [None])[0]
                if filesizes is None or filesize in filesizes:
                    mtime = os.path.getmtime(f)
                    file_infos.append((f, mtime))
            except Exception:
                continue

        latest_files = sorted(file_infos, key=lambda x: x[1], reverse=True)[:max_runs]

        dfs = []
        for f, _ in latest_files:
            try:
                df = pd.read_csv(f)
                df["run_label"] = os.path.basename(f)
                dfs.append(df)
            except Exception:
                continue

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
