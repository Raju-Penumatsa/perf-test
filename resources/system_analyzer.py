# resources/system_analyzer.py

class SystemAnalyzer:
    """
    SystemAnalyzer provides logic to rate each system performance test run
    based on configurable thresholds.
    """

    def __init__(self, thresholds=None):
        """
        Initialize with a dictionary of thresholds to compare against.

        Example:
        {
            "cpu_percent": 90,
            "memory_percent": 90,
            "disk_used_GB": 400
        }
        """
        self.thresholds = thresholds or {}

    def rate_run(self, row) -> str:
        """
        Assign a rating (Good, Warning, Bad) based on how metrics compare to thresholds.

        Returns:
            str: 'Good', 'Warning', or 'Bad'
        """
        warnings = 0
        criticals = 0

        for metric, threshold in self.thresholds.items():
            value = row.get(metric)
            if value is None:
                continue
            try:
                value = float(value)
                if value > threshold * 1.25:
                    criticals += 1
                elif value > threshold:
                    warnings += 1
            except (ValueError, TypeError):
                continue

        if criticals:
            return "Bad"
        elif warnings:
            return "Warning"
        else:
            return "Good"
