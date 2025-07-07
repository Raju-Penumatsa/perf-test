import streamlit as st
from resources.log_manager import LogManager
from resources.system_analyzer import SystemAnalyzer
from resources.anomaly_detector import AnomalyDetector
import plotly.express as px
import pandas as pd
import os
import numpy as np


class DashboardApp:
    """
    DashboardApp encapsulates loading, analyzing, and visualizing
    system performance test results in an interactive Streamlit dashboard.
    """

    def __init__(self, log_dir: str = "logs", max_runs: int = 3):
        """
        Initialize dashboard with log directory and max runs to display.
        """
        self.log_dir = os.path.join(os.path.dirname(__file__), log_dir)
        self.max_runs = max_runs
        self.log_manager = LogManager(log_dir=self.log_dir)
        self.analyzer = SystemAnalyzer(
            thresholds={
                "cpu_percent": 90,
                "memory_percent": 90,
                "disk_used_GB": 400,
            }
        )
        self.df = pd.DataFrame()
        self.time_col = "timestamp"
        self.anomaly_detector = AnomalyDetector(contamination=0.05)

    def configure_page(self):
        """Configure Streamlit page layout and title."""
        st.set_page_config(page_title="System Performance Dashboard", layout="wide")
        st.title("📊 System Performance Dashboard")

    def load_logs(self):
        """
        Load available log files and let the user filter by filesize.
        Returns a DataFrame of filtered logs.
        """
        all_logs_df = self.log_manager.get_log_files_df()
        if all_logs_df.empty:
            st.warning("No log files found in the 'logs/' directory.")
            return None

        st.sidebar.subheader("Available Logs")
        st.sidebar.dataframe(all_logs_df[["filename", "filesize_MB"]])

        filesizes_available = sorted(all_logs_df["filesize_MB"].dropna().unique())
        selected_filesizes = st.sidebar.multiselect(
            "Select Filesize(s) to view trends:",
            filesizes_available,
            default=filesizes_available[:3],
        )

        if not selected_filesizes:
            st.warning("Select at least one filesize in sidebar to view trends.")
            return None

        df = self.log_manager.read_latest_logs(
            filesizes=selected_filesizes, max_runs=self.max_runs
        )
        if df.empty:
            st.warning("No logs could be loaded for selected filesizes.")
            return None

        self.df = df
        self.time_col = "timestamp" if "timestamp" in df.columns else "run_label"
        return df

    def display_summary(self):
        """
        Display host information and summary table of the loaded logs.
        """
        hosts = self.df["hostname"].unique() if "hostname" in self.df.columns else ["Unknown"]
        st.sidebar.write(f"**Host(s) in logs:** {', '.join(hosts)}")

        st.subheader("📋 Combined Summary Table")
        st.dataframe(self.df)

    def display_trends(self):
        st.subheader("📈 Trends (Last Runs)")

        metrics = ["cpu_percent", "memory_used_MB", "disk_used_GB", "read_time_sec", "write_time_sec"]

        # Filter columns actually present
        available_metrics = [m for m in metrics if m in self.df.columns]
        if not available_metrics:
            st.info("No metrics available for plotting.")
            return

        selected_metrics = st.multiselect("Choose metrics to plot:", available_metrics, default=available_metrics)

        if not selected_metrics:
            st.info("Select at least one metric.")
            return

        # Prepare df for plotly
        plot_df = self.df.copy()
        x_axis = self.time_col if self.time_col in self.df.columns else "run_label"

        # Melt dataframe for facet plots
        plot_data = plot_df.melt(
            id_vars=[x_axis, "filesize"],
            value_vars=selected_metrics,
            var_name="Metric",
            value_name="Value"
        )

        fig = px.line(
            plot_data,
            x=x_axis,
            y="Value",
            color="filesize",
            line_dash="Metric",  # Different dash for each metric
            facet_row="Metric",  # One subplot per metric vertically
            markers=True,
            title="System Resource & Test Metrics Over Last Runs",
            labels={x_axis: "Run", "Value": "Metric Value", "filesize": "File Size (MB)"}
        )

        fig.update_layout(height=300 * len(selected_metrics))  # Adjust height by number of metrics

        st.plotly_chart(fig, use_container_width=True)

    def display_ratings(self):
        """
        Apply a rating to each run and show a table with key metrics and performance status.
        """
        st.subheader("🟢🟡🔴 Run Ratings")
        self.df["rating"] = self.df.apply(self.analyzer.rate_run, axis=1)

        preferred_cols = [
            self.time_col,
            "cpu_percent",
            "memory_percent",
            "disk_used_GB",
            "rating",
            "filesize",
        ]
        existing_cols = [c for c in preferred_cols if c in self.df.columns]

        if existing_cols:
            st.dataframe(self.df[existing_cols])
        else:
            st.warning("No expected metric columns found for rating display.")

        missing_cols = set(preferred_cols) - set(self.df.columns)
        if missing_cols:
            st.info(f"⚠️ Missing columns in logs: {', '.join(missing_cols)}")

    def analyze_run_differences(self):
        """
        Compare latest two runs using pandas diff and pct_change to summarize metric changes.
        """
        if self.df.empty or self.time_col not in self.df.columns:
            st.warning("Insufficient data for run analysis.")
            return

        # Select numeric columns only for aggregation
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        if self.time_col in numeric_cols:
            numeric_cols.remove(self.time_col)  # remove time_col if numeric

        # Aggregate metrics per run (mean per run_label or timestamp)
        grouped = self.df.groupby(self.time_col)[numeric_cols].mean()

        if len(grouped) < 2:
            st.info("Need at least two runs for comparison.")
            return

        # Sort runs by time or run label index
        runs_sorted = grouped.sort_index()

        # Calculate differences and percentage changes between current and previous run
        diffs = runs_sorted.diff().iloc[-1]  # last row diff from previous
        pct_changes = runs_sorted.pct_change().iloc[-1] * 100

        prev_label = runs_sorted.index[-2]
        curr_label = runs_sorted.index[-1]

        metrics = [
            "cpu_percent",
            "memory_used_MB",
            "memory_percent",
            "disk_used_GB",
            "read_time_sec",
            "write_time_sec",
        ]

        summary_data = []
        for metric in metrics:
            if metric in runs_sorted.columns:
                prev_val = runs_sorted.iloc[-2][metric]
                curr_val = runs_sorted.iloc[-1][metric]
                diff = diffs.get(metric, 0)
                pct_change = pct_changes.get(metric, 0)

                # Status emoji based on % change
                if pct_change < 0:
                    status = "✅ Improved"
                elif pct_change > 0:
                    status = "⚠️ Worsened"
                else:
                    status = "➖ Unchanged"

                summary_data.append(
                    {
                        "Metric": metric,
                        "Previous Run": round(prev_val, 2),
                        "Current Run": round(curr_val, 2),  # fix here
                        "Difference": round(diff, 2),
                        "% Change": f"{pct_change:.1f}%",
                        "Status": status,
                    }
                )

    def display_enhanced_run_comparison(self):
        """Compare latest two runs with color-coded highlights and advice."""
        if self.df.empty or self.time_col not in self.df.columns:
            st.warning("Insufficient data for run analysis.")
            return

        # Select numeric columns only
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()

        grouped = self.df.groupby(self.time_col)[numeric_cols].mean()

        if len(grouped) < 2:
            st.info("Need at least two runs for comparison.")
            return

        runs_sorted = grouped.sort_index()
        diffs = runs_sorted.diff().iloc[-1]
        pct_changes = runs_sorted.pct_change().iloc[-1] * 100

        prev_label = runs_sorted.index[-2]
        curr_label = runs_sorted.index[-1]

        metrics = [
            "cpu_percent", "memory_used_MB", "memory_percent",
            "disk_used_GB", "read_time_sec", "write_time_sec"
        ]

        summary_data = []
        for metric in metrics:
            if metric in runs_sorted.columns:
                prev_val = runs_sorted.iloc[-2][metric]
                curr_val = runs_sorted.iloc[-1][metric]
                diff = diffs.get(metric, 0)
                pct_change = pct_changes.get(metric, 0)

                if pct_change < -5:
                    status = "✅ Improved"
                    advice = "Performance improved."
                    color = "green"
                elif pct_change > 5:
                    status = "⚠️ Worsened"
                    advice = "Investigate for potential issues."
                    color = "red"
                else:
                    status = "➖ Stable"
                    advice = "No significant change."
                    color = "gray"

                summary_data.append({
                    "Metric": metric,
                    "Previous Run": round(prev_val, 2),
                    "Current Run": round(curr_val, 2),
                    "Difference": round(diff, 2),
                    "% Change": f"{pct_change:.1f}%",
                    "Status": status,
                    "Advice": advice,
                    "Color": color
                })

        summary_df = pd.DataFrame(summary_data)

        st.subheader(f"📊 Run Comparison: {prev_label} vs {curr_label}")

        def highlight_row(row):
            color = summary_df.loc[row.name, "Color"]
            return [f"background-color: {row.Color}"] * len(row)

        # Pass full summary_df (with 'Color' column) to style.apply
        styled_df = summary_df.style.apply(highlight_row, axis=1)

        # Then drop 'Color' column for display by hiding it with CSS or just show all
        st.dataframe(styled_df.hide(axis='columns', subset=['Color']))

    def display_metric_stats(self):
        """Display aggregate statistics (min, max, avg, std) for key metrics."""
        st.subheader("📊 Metric Statistics Across Runs")
        metrics = [
            "cpu_percent", "memory_used_MB", "memory_percent",
            "disk_used_GB", "read_time_sec", "write_time_sec"
        ]
        stats = {}
        for m in metrics:
            if m in self.df.columns:
                stats[m] = {
                    "Min": np.min(self.df[m]),
                    "Max": np.max(self.df[m]),
                    "Average": np.mean(self.df[m]),
                    "Std Dev": np.std(self.df[m])
                }

        if not stats:
            st.info("No metrics available to show statistics.")
            return

        stats_df = pd.DataFrame(stats).T.round(2)
        st.dataframe(stats_df)

    def display_system_metadata(self):
        """Show system metadata summary in sidebar or main panel."""
        st.sidebar.subheader("🖥️ System Metadata")
        if self.df.empty:
            st.sidebar.write("No data available.")
            return

        metadata_keys = [
            "hostname", "os", "processor", "cpu_cores", "cpu_threads",
            "memory_total_MB", "raid_controller"
        ]
        meta = {}
        for key in metadata_keys:
            if key in self.df.columns:
                unique_vals = self.df[key].unique()
                meta[key] = ", ".join(str(v) for v in unique_vals if v)
            else:
                meta[key] = "N/A"

        for k, v in meta.items():
            st.sidebar.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

        # Show last run time
        time_col = self.time_col if self.time_col in self.df.columns else None
        if time_col:
            last_run = self.df[time_col].max()
            st.sidebar.markdown(f"**Last Run:** {last_run}")

    def highlight_anomaly(self, row):
        if row["anomaly"] == -1:
            return ['background-color: #ffcccc'] * len(row)  # Light red background for anomalies
        else:
            return [''] * len(row)  # No highlight for normal rows

    def analyze_anomalies(self):
        if self.df.empty:
            st.info("No data to analyze anomalies.")
            return

        features = [col for col in ["cpu_percent", "memory_used_MB", "disk_used_GB", "read_time_sec", "write_time_sec"]
                    if col in self.df.columns]

        if not features:
            st.warning("No features available for anomaly detection.")
            return

        df_with_anomalies = self.anomaly_detector.detect_anomalies(self.df.copy(), features)

        st.subheader("⚠️ Anomaly Detection Results")
        num_anomalies = (df_with_anomalies["anomaly"] == -1).sum()
        st.write(f"Detected **{num_anomalies}** anomalies in the last {len(df_with_anomalies)} runs.")

        st.write(df_with_anomalies.style.apply(self.highlight_anomaly, axis=1))

    def display_overlay_trends(self):
        st.subheader("📉 Overlay of System Resource Usage Across Runs")

        overlay_metrics = ["cpu_percent", "memory_used_MB", "read_time_sec", "write_time_sec"]
        available = [m for m in overlay_metrics if m in self.df.columns]

        selected = st.multiselect("Select metrics to overlay:", available, default=available[:2])

        if not selected:
            st.warning("Select at least one metric to overlay.")
            return

        for metric in selected:
            fig = px.line(
                self.df,
                x=self.time_col,  # Could also use run index if you want x = 1, 2, 3...
                y=metric,
                color="filesize" if "filesize" in self.df.columns else "run_id",
                line_shape='spline',
                markers=True,
                title=f"Overlay of {metric} Across Runs"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def assign_step_index(self):
        """Add a 'step_index' per run to allow proper line plotting."""
        if "run_label" not in self.df.columns:
            self.df["run_label"] = self.df[self.time_col]
        self.df["step_index"] = self.df.groupby("run_label").cumcount()

    def display_overlay_trends(self):
        st.subheader("📉 Overlay of System Resource Metrics Across Last 3 Runs")

        overlay_metrics = [
            "cpu_percent", "memory_used_MB", "disk_used_GB", "read_time_sec", "write_time_sec"
        ]
        available = [m for m in overlay_metrics if m in self.df.columns]

        selected = st.multiselect("Select metrics to overlay:", available, default=available[:3])

        if not selected:
            st.warning("Select at least one metric to overlay.")
            return

        if "run_label" not in self.df.columns:
            self.df["run_label"] = self.df[self.time_col]

        # Get only last 3 unique run_labels
        last_runs = self.df["run_label"].drop_duplicates().tolist()[-3:]
        df_filtered = self.df[self.df["run_label"].isin(last_runs)]

        for metric in selected:
            if metric not in df_filtered.columns:
                continue

            # Sort for consistent line rendering
            last_runs = self.df["run_label"].drop_duplicates().tail(3)
            df_filtered = self.df[self.df["run_label"].isin(last_runs)]

            fig = px.line(
                df_filtered,
                x="filesize",
                y=metric,
                color="run_label",
                line_group="run_label",
                markers=True,
                title=f"{metric.replace('_', ' ').title()} vs Filesize (Last 3 Runs)",
            )
            fig.update_traces(mode="lines+markers")
            fig.update_layout(
                height=400,
                xaxis_title="Filesize (MB)",
                yaxis_title=metric.replace("_", " ").title()
            )

            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """
        Main method to execute the dashboard lifecycle:
        setup → load logs → display summary → analytics → plot trends → show ratings.
        """
        self.configure_page()
        if self.load_logs() is not None:
            self.display_system_metadata()
            self.display_summary()
            self.display_metric_stats()
            self.display_enhanced_run_comparison()
            self.display_trends()
            self.display_ratings()
            self.analyze_anomalies()
            self.display_overlay_trends()


if __name__ == "__main__":
    app = DashboardApp()
    app.run()
