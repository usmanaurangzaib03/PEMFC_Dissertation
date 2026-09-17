"""
Exploratory data analysis functions for the PEMFC dissertation project.

This module contains reusable statistical functions for examining
individual operational variables in the cleaned PEMFC dataset.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

from typing import Any, Mapping, Sequence
import re
import warnings

from typing import Optional, Sequence, Union
from sklearn.preprocessing import MinMaxScaler

from src.utils import (
    create_safe_filename,
    ensure_directory,
    save_dataframe,
)


def format_statistics_table(statistics, unit=""):
    """Return a readable, display-formatted copy of a statistics table."""

    if not isinstance(statistics, pd.DataFrame):
        raise TypeError("statistics must be a pandas DataFrame.")

    required_columns = {"Metric", "Value"}
    if not required_columns.issubset(statistics.columns):
        raise ValueError(
            "statistics must contain 'Metric' and 'Value' columns."
        )

    formatted = statistics.copy()

    count_metrics = {
        "Total Observations",
        "Valid Observations",
        "Missing Values",
        "Unique Values",
        "IQR Outlier Count",
    }

    percentage_metrics = {
        "Missing (%)",
        "Coefficient of Variation (%)",
        "IQR Outliers (%)",
    }

    dimensionless_metrics = {
        "Skewness",
        "Kurtosis",
    }

    def format_row(row):
        metric = row["Metric"]
        value = row["Value"]

        if pd.isna(value):
            return ""

        if metric in count_metrics:
            return f"{int(value):,}"

        if metric == "Missing (%)":
            return f"{float(value):.2f}%"

        if metric == "Coefficient of Variation (%)":
            return f"{float(value):.2f}%"

        if metric == "IQR Outliers (%)":
            # Retain enough precision because the percentage may be very small.
            return f"{float(value):.6f}%"

        if metric in dimensionless_metrics:
            return f"{float(value):.4f}"

        if metric == "Variance":
            formatted_value = f"{float(value):.6f}"
            return f"{formatted_value} {unit}²" if unit else formatted_value

        formatted_value = f"{float(value):.4f}"
        return f"{formatted_value} {unit}" if unit else formatted_value

    formatted["Value"] = formatted.apply(format_row, axis=1)
    return formatted


def variable_summary(
    dataframe,
    variable,
    description="",
    unit="",
    format_values=True,
):
    """Calculate metadata and descriptive statistics for one numeric variable."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("The supplied dataframe must be a pandas DataFrame.")

    if variable not in dataframe.columns:
        raise KeyError(f"Variable '{variable}' was not found in the dataset.")

    if not pd.api.types.is_numeric_dtype(dataframe[variable]):
        raise TypeError(f"Variable '{variable}' must contain numerical data.")

    series = dataframe[variable]
    valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()

    if valid_series.empty:
        raise ValueError(
            f"Variable '{variable}' contains no valid numerical observations."
        )

    total_observations = len(series)
    valid_observations = valid_series.shape[0]
    missing_values = total_observations - valid_observations
    missing_percentage = (missing_values / total_observations) * 100
    unique_values = valid_series.nunique()

    mean_value = valid_series.mean()
    median_value = valid_series.median()
    mode_values = valid_series.mode()
    mode_value = np.nan if mode_values.empty else mode_values.iloc[0]

    minimum_value = valid_series.min()
    q1_value = valid_series.quantile(0.25)
    q3_value = valid_series.quantile(0.75)
    maximum_value = valid_series.max()
    range_value = maximum_value - minimum_value

    variance_value = valid_series.var()
    standard_deviation = valid_series.std()
    iqr_value = q3_value - q1_value

    if np.isclose(mean_value, 0):
        coefficient_of_variation = np.nan
    else:
        coefficient_of_variation = (
            standard_deviation / abs(mean_value)
        ) * 100

    skewness_value = valid_series.skew()
    kurtosis_value = valid_series.kurt()

    lower_outlier_bound = q1_value - (1.5 * iqr_value)
    upper_outlier_bound = q3_value + (1.5 * iqr_value)

    outlier_mask = (
        (valid_series < lower_outlier_bound)
        | (valid_series > upper_outlier_bound)
    )
    outlier_count = int(outlier_mask.sum())
    outlier_percentage = (outlier_count / valid_observations) * 100

    metadata = pd.DataFrame(
        {
            "Property": ["Variable", "Description", "Unit", "Data Type"],
            "Value": [variable, description, unit, str(series.dtype)],
        }
    )

    statistics = pd.DataFrame(
        {
            "Metric": [
                "Total Observations",
                "Valid Observations",
                "Missing Values",
                "Missing (%)",
                "Unique Values",
                "Minimum",
                "Q1",
                "Mean",
                "Median",
                "Mode",
                "Q3",
                "Maximum",
                "Range",
                "Variance",
                "Standard Deviation",
                "IQR",
                "Coefficient of Variation (%)",
                "Skewness",
                "Kurtosis",
                "IQR Lower Bound",
                "IQR Upper Bound",
                "IQR Outlier Count",
                "IQR Outliers (%)",
            ],
            "Value": [
                total_observations,
                valid_observations,
                missing_values,
                missing_percentage,
                unique_values,
                minimum_value,
                q1_value,
                mean_value,
                median_value,
                mode_value,
                q3_value,
                maximum_value,
                range_value,
                variance_value,
                standard_deviation,
                iqr_value,
                coefficient_of_variation,
                skewness_value,
                kurtosis_value,
                lower_outlier_bound,
                upper_outlier_bound,
                outlier_count,
                outlier_percentage,
            ],
        }
    )

    if format_values:
        statistics = format_statistics_table(statistics, unit=unit)

    return metadata, statistics


# ---------------------------------------------------------------------------
# Variable profile
# ---------------------------------------------------------------------------


def variable_profile(
    dataframe,
    variable,
    description="",
    unit="",
    bins=50,
    output_directory=None,
    save_outputs=False,
    show_plots=True,
):
    """Generate a summary, histogram, and boxplot for one numeric variable."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if not isinstance(variable, str) or not variable.strip():
        raise ValueError("variable must be a non-empty string.")

    if variable not in dataframe.columns:
        raise KeyError(f"Variable '{variable}' was not found in the DataFrame.")

    if not pd.api.types.is_numeric_dtype(dataframe[variable]):
        raise TypeError(f"Variable '{variable}' must contain numerical data.")

    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError("bins must be a positive integer.")

    series = (
        dataframe[variable]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if series.empty:
        raise ValueError(
            f"Variable '{variable}' contains no valid finite observations."
        )

    metadata, statistics = variable_summary(
        dataframe=dataframe,
        variable=variable,
        description=description,
        unit=unit,
        format_values=True,
    )

    variable_label = variable.replace("_", " ").title()
    axis_label = f"{variable_label} ({unit})" if unit else variable_label

    # Histogram: raw observation counts, not probability density.
    histogram_figure, histogram_axis = plt.subplots(figsize=(10, 6))
    histogram_axis.hist(
        series,
        bins=bins,
        density=False,
        edgecolor="black",
        alpha=0.85,
    )

    mean_value = series.mean()
    median_value = series.median()
    unit_suffix = f" {unit}" if unit else ""

    histogram_axis.axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean (μ) = {mean_value:.4f}{unit_suffix}",
    )
    histogram_axis.axvline(
        median_value,
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {median_value:.4f}{unit_suffix}",
    )

    histogram_axis.set_title(f"Distribution of {variable_label}")
    histogram_axis.set_xlabel(axis_label)
    histogram_axis.set_ylabel("Number of Observations")
    histogram_axis.legend()
    histogram_axis.grid(axis="y", alpha=0.3)
    histogram_figure.tight_layout()

    # Separate horizontal boxplot for flexible dissertation use.
    boxplot_figure, boxplot_axis = plt.subplots(figsize=(10, 4))
    boxplot_axis.boxplot(
        series,
        orientation="horizontal",
        showmeans=True,
        meanline=False,
        notch=False,
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "none",
        },
    )

    boxplot_axis.set_title(f"Boxplot of {variable_label}")
    boxplot_axis.set_xlabel(axis_label)
    boxplot_axis.set_yticks([])
    boxplot_axis.grid(axis="x", alpha=0.3)
    boxplot_figure.tight_layout()

    metadata_path = None
    statistics_path = None
    histogram_path = None
    boxplot_path = None

    if save_outputs:
        if output_directory is None:
            raise ValueError(
                "output_directory must be provided when save_outputs=True."
            )

        output_directory = ensure_directory(output_directory)
        safe_variable_name = create_safe_filename(variable)

        metadata_path = output_directory / (
            f"variable_metadata_{safe_variable_name}.csv"
        )
        statistics_path = output_directory / (
            f"variable_statistics_{safe_variable_name}.csv"
        )
        histogram_path = output_directory / (
            f"histogram_{safe_variable_name}.png"
        )
        boxplot_path = output_directory / (
            f"boxplot_{safe_variable_name}.png"
        )

        save_dataframe(metadata, metadata_path, index=False)
        save_dataframe(statistics, statistics_path, index=False)

        histogram_figure.savefig(
            histogram_path,
            dpi=300,
            bbox_inches="tight",
        )
        boxplot_figure.savefig(
            boxplot_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show_plots:
        plt.show()
    else:
        plt.close(histogram_figure)
        plt.close(boxplot_figure)

    return {
        "metadata": metadata,
        "statistics": statistics,
        "histogram_figure": histogram_figure,
        "boxplot_figure": boxplot_figure,
        "metadata_path": metadata_path,
        "statistics_path": statistics_path,
        "histogram_path": histogram_path,
        "boxplot_path": boxplot_path,
    }
#-----------------------------------------------------------------------------Noteboob 8 - Reusable Functions --------------------------------------------------------------------------------
# ============================================================
# Time-Series Integrity Assessment
# ============================================================

def assess_time_integrity(
    df,
    time_column="time",
    group_column="operating_hour",
    gap_multiplier=2.0
):
    """
    Assess time-series integrity within each experimental group.

    The function evaluates chronological ordering, missing time values,
    duplicate timestamps, sampling intervals, and large time gaps
    separately within each operating-hour experiment.

    Parameters
    ----------
    df : pandas.DataFrame
        Input time-series dataset.

    time_column : str, default="time"
        Name of the local time variable.

    group_column : str, default="operating_hour"
        Name of the column identifying separate experiments or stages.

    gap_multiplier : float, default=2.0
        A sampling interval greater than the group median multiplied by
        this value is classified as a large time gap.

    Returns
    -------
    overall_summary : pandas.DataFrame
        Concise overall time-series integrity summary.

    experiment_summary : pandas.DataFrame
        Detailed integrity assessment for every experimental group.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The input dataset is empty.")

    required_columns = [group_column, time_column]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required column(s) not found in the dataset: "
            f"{missing_columns}"
        )

    if gap_multiplier <= 0:
        raise ValueError("gap_multiplier must be greater than zero.")

    # --------------------------------------------------------
    # Assess each experimental group
    # --------------------------------------------------------

    experiment_results = []

    grouped_data = df.groupby(
        group_column,
        sort=True,
        dropna=False
    )

    for group_value, group_df in grouped_data:

        time_values = pd.to_numeric(
            group_df[time_column],
            errors="coerce"
        )

        missing_time_values = int(
            time_values.isna().sum()
        )

        valid_time = time_values.dropna()

        duplicate_timestamps = int(
            valid_time.duplicated().sum()
        )

        chronological_order = bool(
            valid_time.is_monotonic_increasing
        )

        intervals = valid_time.diff().dropna()

        if intervals.empty:
            mean_interval = np.nan
            median_interval = np.nan
            minimum_interval = np.nan
            maximum_interval = np.nan
            interval_std = np.nan
            large_time_gaps = 0
            expected_interval = np.nan
        else:
            mean_interval = intervals.mean()
            median_interval = intervals.median()
            minimum_interval = intervals.min()
            maximum_interval = intervals.max()
            interval_std = intervals.std()

            expected_interval = median_interval

            large_time_gaps = int(
                (
                    intervals >
                    expected_interval * gap_multiplier
                ).sum()
            )

        if valid_time.empty:
            start_time = np.nan
            end_time = np.nan
            duration = np.nan
        else:
            start_time = valid_time.iloc[0]
            end_time = valid_time.iloc[-1]
            duration = end_time - start_time

        experiment_results.append({
            "Operating Hour": group_value,
            "Observations": len(group_df),
            "Missing Time Values": missing_time_values,
            "Duplicate Timestamps": duplicate_timestamps,
            "Chronological Order": chronological_order,
            "Start Time": start_time,
            "End Time": end_time,
            "Duration": duration,
            "Mean Interval": mean_interval,
            "Median Interval": median_interval,
            "Minimum Interval": minimum_interval,
            "Maximum Interval": maximum_interval,
            "Interval Standard Deviation": interval_std,
            "Expected Interval": expected_interval,
            "Large Time Gaps": large_time_gaps
        })

    experiment_summary = pd.DataFrame(
        experiment_results
    )

    # --------------------------------------------------------
    # Overall assessment
    # --------------------------------------------------------

    total_missing_time_values = int(
        experiment_summary[
            "Missing Time Values"
        ].sum()
    )

    total_duplicate_timestamps = int(
        experiment_summary[
            "Duplicate Timestamps"
        ].sum()
    )

    all_groups_chronological = bool(
        experiment_summary[
            "Chronological Order"
        ].all()
    )

    total_large_time_gaps = int(
        experiment_summary[
            "Large Time Gaps"
        ].sum()
    )

    experiments_with_ordering_issues = int(
        (
            ~experiment_summary[
                "Chronological Order"
            ]
        ).sum()
    )

    experiments_with_duplicate_timestamps = int(
        (
            experiment_summary[
                "Duplicate Timestamps"
            ] > 0
        ).sum()
    )

    experiments_with_large_gaps = int(
        (
            experiment_summary[
                "Large Time Gaps"
            ] > 0
        ).sum()
    )

    integrity_passed = (
        total_missing_time_values == 0
        and total_duplicate_timestamps == 0
        and all_groups_chronological
        and total_large_time_gaps == 0
    )

    overall_summary = pd.DataFrame({
        "Assessment": [
            "Time Column",
            "Grouping Column",
            "Total Observations",
            "Number of Experiments",
            "Missing Time Values",
            "Duplicate Timestamps Within Experiments",
            "Experiments with Duplicate Timestamps",
            "All Experiments Chronologically Ordered",
            "Experiments with Ordering Issues",
            "Overall Mean Sampling Interval",
            "Overall Median Sampling Interval",
            "Minimum Sampling Interval",
            "Maximum Sampling Interval",
            "Large Time Gaps",
            "Experiments Containing Large Gaps",
            "Time-Series Integrity Status"
        ],
        "Result": [
            time_column,
            group_column,
            f"{len(df):,}",
            experiment_summary.shape[0],
            total_missing_time_values,
            total_duplicate_timestamps,
            experiments_with_duplicate_timestamps,
            (
                "Yes"
                if all_groups_chronological
                else "No"
            ),
            experiments_with_ordering_issues,
            experiment_summary[
                "Mean Interval"
            ].mean(),
            experiment_summary[
                "Median Interval"
            ].median(),
            experiment_summary[
                "Minimum Interval"
            ].min(),
            experiment_summary[
                "Maximum Interval"
            ].max(),
            total_large_time_gaps,
            experiments_with_large_gaps,
            (
                "Passed — Ready for Behaviour Exploration"
                if integrity_passed
                else "Review Required"
            )
        ]
    })

    return overall_summary, experiment_summary

# ============================================================
# Extract Representative Time-Series Window
# ============================================================

def extract_timeseries_window(
    df,
    stage,
    variable,
    group_column="operating_hour",
    time_column="time",
    window_start=0,
    window_length=1000
):
    """
    Extract a continuous time-series window from one experimental stage.

    Parameters
    ----------
    df : pandas.DataFrame
        Input operational dataset.

    stage : int or float
        Operating-hour experiment to extract.

    variable : str
        Variable to include in the returned window.

    group_column : str, default="operating_hour"
        Column identifying experimental stages.

    time_column : str, default="time"
        Local time variable.

    window_start : int, default=0
        Starting row position within the selected stage.

    window_length : int, default=1000
        Number of consecutive observations to extract.

    Returns
    -------
    pandas.DataFrame
        Continuous time-series window.
    """

    import pandas as pd

    required_columns = [
        group_column,
        time_column,
        variable
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required column(s) not found: {missing_columns}"
        )

    if window_start < 0:
        raise ValueError("window_start cannot be negative.")

    if window_length <= 0:
        raise ValueError("window_length must be greater than zero.")

    stage_data = (
        df.loc[
            df[group_column].eq(stage),
            [group_column, time_column, variable]
        ]
        .sort_values(time_column)
        .reset_index(drop=True)
    )

    if stage_data.empty:
        raise ValueError(
            f"No observations found for operating hour {stage}."
        )

    window_end = window_start + window_length

    if window_start >= len(stage_data):
        raise ValueError(
            f"window_start exceeds the available observations "
            f"for operating hour {stage}."
        )

    return stage_data.iloc[
        window_start:window_end
    ].copy()

#---------------------------------------------------------------------------------------------------
# ============================================================
# Plot Stage-Level Variable Behaviour
# ============================================================

def plot_stage_behaviour(
    df,
    variable,
    group_column="operating_hour",
    ylabel=None,
    title=None,
    figsize=(14, 5)
):
    """
    Summarise and plot stage-level behaviour using the median
    and interquartile range.

    The median represents the typical operating value within each
    durability stage. The interquartile range represents the middle
    50% of observations and reduces the visual influence of isolated
    extreme values.

    Minimum and maximum values are retained in the returned summary
    table but are not plotted.

    Parameters
    ----------
    df : pandas.DataFrame
        Input operational dataset.

    variable : str
        Numeric variable to analyse.

    group_column : str, default="operating_hour"
        Column identifying the durability stages.

    ylabel : str or None, default=None
        Label used for the y-axis.

    title : str or None, default=None
        Figure title.

    figsize : tuple, default=(14, 5)
        Figure dimensions.

    Returns
    -------
    summary : pandas.DataFrame
        Stage-level table containing the median, quartiles, IQR,
        minimum, and maximum.

    figure : matplotlib.figure.Figure
        Stage-level behaviour figure.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The input dataset is empty.")

    required_columns = [group_column, variable]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required column(s) not found: {missing_columns}"
        )

    if not pd.api.types.is_numeric_dtype(df[variable]):
        raise TypeError(
            f"'{variable}' must be numeric."
        )

    # --------------------------------------------------------
    # Calculate stage-level behaviour statistics
    # --------------------------------------------------------

    summary = (
        df
        .groupby(group_column, sort=True)[variable]
        .agg(
            Median="median",
            Q1=lambda values: values.quantile(0.25),
            Q3=lambda values: values.quantile(0.75),
            Minimum="min",
            Maximum="max"
        )
        .reset_index()
    )

    summary["IQR"] = (
        summary["Q3"] - summary["Q1"]
    )

    summary = summary[
        [
            group_column,
            "Median",
            "Q1",
            "Q3",
            "IQR",
            "Minimum",
            "Maximum"
        ]
    ]

    # --------------------------------------------------------
    # Create stage-level figure
    # --------------------------------------------------------

    display_label = (
        ylabel
        if ylabel is not None
        else variable.replace("_", " ").title()
    )

    figure, axis = plt.subplots(figsize=figsize)

    axis.plot(
        summary[group_column],
        summary["Median"],
        marker="o",
        linewidth=2,
        label="Stage median"
    )

    axis.fill_between(
        summary[group_column],
        summary["Q1"],
        summary["Q3"],
        alpha=0.25,
        label="Interquartile range (IQR)"
    )

    axis.set_title(
        title
        if title is not None
        else f"{display_label} Behaviour Across Durability Stages"
    )

    axis.set_xlabel("Operating Hour")
    axis.set_ylabel(display_label)

    axis.grid(True, alpha=0.30)
    axis.legend()

    figure.tight_layout()

    return summary, figure



# ============================================================
# Plot Representative Stage Windows
# ============================================================

def plot_representative_windows(
    df,
    variable,
    selected_stages,
    group_column="operating_hour",
    time_column="time",
    window_start=0,
    window_length=1000,
    ylabel=None,
    title=None,
    figsize=(14, 7)
):
    """
    Plot continuous representative time windows for selected stages.

    Parameters
    ----------
    df : pandas.DataFrame
        Input operational dataset.

    variable : str
        Variable to plot.

    selected_stages : list
        Operating-hour experiments to compare.

    group_column : str, default="operating_hour"
        Column identifying experimental stages.

    time_column : str, default="time"
        Local time variable.

    window_start : int, default=0
        Starting row position within each stage.

    window_length : int, default=1000
        Number of consecutive observations plotted per stage.

    ylabel : str or None
        Y-axis label.

    title : str or None
        Main figure title.

    figsize : tuple, default=(14, 7)
        Figure size.

    Returns
    -------
    tuple
        windows : dict
            Extracted windows by stage.

        figure : matplotlib.figure.Figure
            Generated figure.
    """

    import matplotlib.pyplot as plt

    display_label = (
        ylabel
        or variable.replace("_", " ").title()
    )

    windows = {}

    figure, axis = plt.subplots(figsize=figsize)

    for stage in selected_stages:

        window = extract_timeseries_window(
            df=df,
            stage=stage,
            variable=variable,
            group_column=group_column,
            time_column=time_column,
            window_start=window_start,
            window_length=window_length
        )

        windows[stage] = window

        relative_time = (
            window[time_column]
            - window[time_column].iloc[0]
        )

        axis.plot(
            relative_time,
            window[variable],
            label=f"{stage} h"
        )

    axis.set_title(
        title
        or f"Representative {display_label} Behaviour"
    )

    axis.set_xlabel(
        "Time Within Selected Window (s)"
    )

    axis.set_ylabel(display_label)
    axis.grid(True, alpha=0.30)
    axis.legend(title="Durability Stage")

    figure.tight_layout()

    return windows, figure

from sklearn.preprocessing import MinMaxScaler
#------------------------------------------------------------------------------------

# ==========================================================
# Complete Subsystem Behaviour Analysis
# ==========================================================

def analyse_subsystem_behaviour(
    df: pd.DataFrame,
    variables: Sequence[str],
    subsystem_name: str,
    stage_column: str = "operating_hour",
    time_column: str = "time",
    labels: Optional[Sequence[str]] = None,
    representative_stages: Sequence[Union[int, float]] = (50, 550, 1000),
    representative_points: Optional[int] = 5000,
    representative_start: int = 0,
    variability_measure: str = "iqr",
    decimal_places: int = 4,
    save_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = True,
) -> dict:
    """
    Perform complete subsystem behaviour analysis.

    The function implements the locked subsystem-analysis framework:

    1. Stage-Level Normalised Behaviour
       - Applies Min-Max normalisation independently to each variable.
       - Calculates the stage-level median.
       - Compares long-term temporal evolution and behavioural trends.

    2. Stage-Level Variability Analysis
       - Calculates Q1, median, Q3, IQR and standard deviation.
       - Evaluates behavioural stability and variability across stages.

    3. Representative Operating Behaviour
       - Extracts selected early, middle and late operating stages.
       - Compares short-term fluctuations and operating patterns.
       - Supports analysis of synchronisation and repeatability.

    Parameters
    ----------
    df : pandas.DataFrame
        Operational dataframe containing all required columns.
        The original dataframe is not modified.

    variables : sequence of str
        Subsystem variables to analyse.

    subsystem_name : str
        Name used in plot titles and saved filenames.
        Examples: "Electrical", "Pressure", "Temperature",
        or "Reactant Flow".

    stage_column : str, default="operating_hour"
        Column identifying the durability stage.

    time_column : str, default="time"
        Column representing time within each stage.

    labels : sequence of str, optional
        Readable labels corresponding to the variables.
        When omitted, labels are generated from variable names.

    representative_stages : sequence, default=(50, 550, 1000)
        Durability stages selected for representative operating analysis.

    representative_points : int or None, default=5000
        Maximum number of consecutive observations extracted from each
        representative stage. When None, the complete stage is used.

    representative_start : int, default=0
        Starting row position within each representative stage.

    variability_measure : {"iqr", "std"}, default="iqr"
        Statistic displayed in the variability comparison plot.

    decimal_places : int, default=4
        Number of decimal places used in presentation-ready tables.

    save_dir : str or pathlib.Path, optional
        Directory in which figures and tables are saved.
        Files are not saved when this is None.

    show_plots : bool, default=True
        Whether plots are displayed in the notebook.

    Returns
    -------
    dict
        Dictionary containing:

        Presentation-ready outputs
        --------------------------
        stage_central_summary
        stage_variability_summary

        Analytical outputs
        ------------------
        stage_central_summary_raw
        stage_variability_summary_raw
        representative_data
        normalised_data
        normalised_columns
        scaler

        Saved output information
        ------------------------
        figure_paths
        table_paths

    Notes
    -----
    Min-Max normalisation is performed independently for each variable:

        x_normalised = (x - x_min) / (x_max - x_min)

    The normalised values are used only for comparison. They do not
    represent the original engineering units.
    """

    # ------------------------------------------------------
    # 1. Validate main inputs
    # ------------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The supplied dataframe is empty.")

    if not isinstance(subsystem_name, str) or not subsystem_name.strip():
        raise ValueError("subsystem_name must be a non-empty string.")

    variables = list(variables)

    if len(variables) == 0:
        raise ValueError(
            "At least one subsystem variable must be supplied."
        )

    if len(set(variables)) != len(variables):
        raise ValueError(
            "The variables sequence contains duplicate variable names."
        )

    required_columns = [
        stage_column,
        time_column,
        *variables,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "The following required columns are missing from the "
            f"dataframe: {missing_columns}"
        )

    # ------------------------------------------------------
    # 2. Validate numeric subsystem variables
    # ------------------------------------------------------
    non_numeric_variables = [
        variable
        for variable in variables
        if not pd.api.types.is_numeric_dtype(df[variable])
    ]

    if non_numeric_variables:
        raise TypeError(
            "All subsystem variables must be numeric. "
            f"Non-numeric variables: {non_numeric_variables}"
        )

    missing_value_counts = df[variables].isna().sum()
    missing_value_counts = missing_value_counts[
        missing_value_counts > 0
    ]

    if not missing_value_counts.empty:
        raise ValueError(
            "Subsystem variables contain missing values. "
            f"Missing-value counts: {missing_value_counts.to_dict()}"
        )

    infinite_variables = [
        variable
        for variable in variables
        if np.isinf(df[variable].to_numpy()).any()
    ]

    if infinite_variables:
        raise ValueError(
            "Subsystem variables contain infinite values: "
            f"{infinite_variables}"
        )

    # ------------------------------------------------------
    # 3. Prepare readable labels
    # ------------------------------------------------------
    if labels is None:
        labels = [
            variable.replace("_", " ").title()
            for variable in variables
        ]
    else:
        labels = list(labels)

    if len(labels) != len(variables):
        raise ValueError(
            "The number of labels must match the number of variables."
        )

    if len(set(labels)) != len(labels):
        raise ValueError(
            "Each subsystem variable must have a unique display label."
        )

    variable_to_label = dict(zip(variables, labels))

    normalised_columns = [
        f"{variable}_normalised"
        for variable in variables
    ]

    normalised_to_label = dict(
        zip(normalised_columns, labels)
    )

    # ------------------------------------------------------
    # 4. Validate analysis settings
    # ------------------------------------------------------
    representative_stages = list(representative_stages)

    if len(representative_stages) == 0:
        raise ValueError(
            "At least one representative stage must be supplied."
        )

    if representative_start < 0:
        raise ValueError(
            "representative_start cannot be negative."
        )

    if (
        representative_points is not None
        and representative_points <= 0
    ):
        raise ValueError(
            "representative_points must be greater than zero or None."
        )

    variability_measure = variability_measure.lower().strip()

    if variability_measure not in {"iqr", "std"}:
        raise ValueError(
            "variability_measure must be either 'iqr' or 'std'."
        )

    if not isinstance(decimal_places, int) or decimal_places < 0:
        raise ValueError(
            "decimal_places must be a non-negative integer."
        )

    # ------------------------------------------------------
    # 5. Create protected working dataframe
    # ------------------------------------------------------
    subsystem_df = df[
        [
            stage_column,
            time_column,
            *variables,
        ]
    ].copy()

    subsystem_df = (
        subsystem_df
        .sort_values(
            by=[stage_column, time_column]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # 6. Apply Min-Max normalisation
    # ------------------------------------------------------
    scaler = MinMaxScaler()

    subsystem_df[normalised_columns] = scaler.fit_transform(
        subsystem_df[variables]
    )

    # Identify variables that are constant across the dataset.
    # MinMaxScaler assigns zero to their normalised values.
    constant_variables = [
        variable
        for variable in variables
        if subsystem_df[variable].nunique(dropna=True) == 1
    ]

    if constant_variables:
        print(
            "Note: The following variables were constant across the "
            "supplied dataframe and therefore received normalised "
            f"values of 0: {constant_variables}"
        )

    # ------------------------------------------------------
    # 7. Calculate stage-level central tendency
    # ------------------------------------------------------
    stage_central_summary_raw = (
        subsystem_df
        .groupby(
            stage_column,
            as_index=False,
            sort=True,
        )[normalised_columns]
        .median()
        .sort_values(stage_column)
        .reset_index(drop=True)
    )

    # Create presentation-ready central summary.
    central_column_mapping = {
        stage_column: "Operating Hour",
        **normalised_to_label,
    }

    stage_central_summary = (
        stage_central_summary_raw
        .rename(columns=central_column_mapping)
        .round(decimal_places)
    )

    # ------------------------------------------------------
    # 8. Calculate stage-level variability statistics
    # ------------------------------------------------------
    grouped_data = subsystem_df.groupby(
        stage_column,
        sort=True,
    )[normalised_columns]

    q1_summary = grouped_data.quantile(0.25)
    median_summary = grouped_data.median()
    q3_summary = grouped_data.quantile(0.75)
    standard_deviation_summary = grouped_data.std(ddof=1)
    observation_summary = grouped_data.count()

    iqr_summary = q3_summary - q1_summary

    variability_frames = []

    for variable, normalised_column in zip(
        variables,
        normalised_columns,
    ):
        variable_summary = pd.DataFrame(
            {
                stage_column: median_summary.index,
                "variable": variable,
                "label": variable_to_label[variable],
                "q1": q1_summary[
                    normalised_column
                ].to_numpy(),
                "median": median_summary[
                    normalised_column
                ].to_numpy(),
                "q3": q3_summary[
                    normalised_column
                ].to_numpy(),
                "iqr": iqr_summary[
                    normalised_column
                ].to_numpy(),
                "std": standard_deviation_summary[
                    normalised_column
                ].to_numpy(),
                "count": observation_summary[
                    normalised_column
                ].to_numpy(),
            }
        )

        variability_frames.append(variable_summary)

    stage_variability_summary_raw = pd.concat(
        variability_frames,
        ignore_index=True,
    )

    stage_variability_summary_raw = (
        stage_variability_summary_raw
        .sort_values(
            by=[stage_column, "variable"]
        )
        .reset_index(drop=True)
    )

    # Create presentation-ready variability summary.
    variability_column_mapping = {
        stage_column: "Operating Hour",
        "label": "Variable",
        "q1": "Q1",
        "median": "Median",
        "q3": "Q3",
        "iqr": "IQR",
        "std": "Standard Deviation",
        "count": "Observations",
    }

    stage_variability_summary = (
        stage_variability_summary_raw[
            [
                stage_column,
                "label",
                "q1",
                "median",
                "q3",
                "iqr",
                "std",
                "count",
            ]
        ]
        .rename(columns=variability_column_mapping)
    )

    numeric_variability_columns = [
        "Q1",
        "Median",
        "Q3",
        "IQR",
        "Standard Deviation",
    ]

    stage_variability_summary[
        numeric_variability_columns
    ] = stage_variability_summary[
        numeric_variability_columns
    ].round(decimal_places)

    # Ensure the observation count remains an integer.
    stage_variability_summary[
        "Observations"
    ] = stage_variability_summary[
        "Observations"
    ].astype(int)

    # ------------------------------------------------------
    # 9. Extract representative operating stages
    # ------------------------------------------------------
    available_stages = set(
        subsystem_df[stage_column].unique()
    )

    unavailable_stages = [
        stage
        for stage in representative_stages
        if stage not in available_stages
    ]

    if unavailable_stages:
        raise ValueError(
            "The following representative stages are unavailable: "
            f"{unavailable_stages}. Available stages are: "
            f"{sorted(available_stages)}"
        )

    representative_frames = []

    for stage in representative_stages:
        stage_data = (
            subsystem_df.loc[
                subsystem_df[stage_column] == stage
            ]
            .sort_values(time_column)
            .reset_index(drop=True)
        )

        if representative_points is None:
            stage_data = stage_data.iloc[
                representative_start:
            ].copy()
        else:
            stage_data = stage_data.iloc[
                representative_start:
                representative_start + representative_points
            ].copy()

        if stage_data.empty:
            raise ValueError(
                f"No observations were extracted for stage {stage}. "
                "Review representative_start and "
                "representative_points."
            )

        # Produce a relative time axis for direct stage comparison.
        try:
            stage_data["relative_time"] = (
                stage_data[time_column]
                - stage_data[time_column].iloc[0]
            )
        except (TypeError, ValueError):
            # Use observation index when the time column cannot
            # be subtracted safely.
            stage_data["relative_time"] = np.arange(
                len(stage_data)
            )

        stage_data["representative_stage"] = stage

        representative_frames.append(stage_data)

    representative_data = pd.concat(
        representative_frames,
        ignore_index=True,
    )

    # ------------------------------------------------------
    # 10. Prepare output directories
    # ------------------------------------------------------
    figure_paths = {}
    table_paths = {}

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_subsystem_name = (
            subsystem_name
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    # ------------------------------------------------------
    # 11. Plot stage-level normalised behaviour
    # ------------------------------------------------------
    stage_figure, stage_axis = plt.subplots(
        figsize=(11, 6)
    )

    for normalised_column in normalised_columns:
        stage_axis.plot(
            stage_central_summary_raw[stage_column],
            stage_central_summary_raw[normalised_column],
            marker="o",
            linewidth=2,
            label=normalised_to_label[normalised_column],
        )

    stage_axis.set_title(
        f"Stage-Level Normalised {subsystem_name} Behaviour"
    )
    stage_axis.set_xlabel("Operating Hour")
    stage_axis.set_ylabel("Normalised Median")
    stage_axis.set_ylim(-0.05, 1.05)
    stage_axis.grid(True, alpha=0.3)
    stage_axis.legend()
    stage_figure.tight_layout()

    if save_dir is not None:
        stage_figure_path = (
            save_dir
            / (
                f"{safe_subsystem_name}_"
                "stage_level_normalised_behaviour.png"
            )
        )

        stage_figure.savefig(
            stage_figure_path,
            dpi=300,
            bbox_inches="tight",
        )

        figure_paths[
            "stage_level_normalised_behaviour"
        ] = stage_figure_path

    if show_plots:
        plt.show()
    else:
        plt.close(stage_figure)

    # ------------------------------------------------------
    # 12. Plot stage-level variability
    # ------------------------------------------------------
    variability_figure, variability_axis = plt.subplots(
        figsize=(11, 6)
    )

    variability_axis_label = (
        "Normalised Interquartile Range (IQR)"
        if variability_measure == "iqr"
        else "Normalised Standard Deviation"
    )

    for variable, label in zip(variables, labels):
        variable_variability = (
            stage_variability_summary_raw.loc[
                stage_variability_summary_raw[
                    "variable"
                ] == variable
            ]
        )

        variability_axis.plot(
            variable_variability[stage_column],
            variable_variability[variability_measure],
            marker="o",
            linewidth=2,
            label=label,
        )

    variability_axis.set_title(
        f"Stage-Level {subsystem_name} Variability"
    )
    variability_axis.set_xlabel("Operating Hour")
    variability_axis.set_ylabel(
        variability_axis_label
    )
    variability_axis.grid(True, alpha=0.3)
    variability_axis.legend()
    variability_figure.tight_layout()

    if save_dir is not None:
        variability_figure_path = (
            save_dir
            / (
                f"{safe_subsystem_name}_"
                "stage_level_variability.png"
            )
        )

        variability_figure.savefig(
            variability_figure_path,
            dpi=300,
            bbox_inches="tight",
        )

        figure_paths[
            "stage_level_variability"
        ] = variability_figure_path

    if show_plots:
        plt.show()
    else:
        plt.close(variability_figure)

    # ------------------------------------------------------
    # 13. Plot representative operating behaviour
    # ------------------------------------------------------
    for stage in representative_stages:
        stage_representative_data = (
            representative_data.loc[
                representative_data[
                    "representative_stage"
                ] == stage
            ]
        )

        representative_figure, representative_axis = (
            plt.subplots(figsize=(12, 6))
        )

        for normalised_column in normalised_columns:
            representative_axis.plot(
                stage_representative_data[
                    "relative_time"
                ],
                stage_representative_data[
                    normalised_column
                ],
                linewidth=1.2,
                label=normalised_to_label[
                    normalised_column
                ],
            )

        representative_axis.set_title(
            f"Representative {subsystem_name} "
            f"Operating Behaviour at {stage} h"
        )
        representative_axis.set_xlabel("Relative Time")
        representative_axis.set_ylabel("Normalised Value")
        representative_axis.set_ylim(-0.05, 1.05)
        representative_axis.grid(True, alpha=0.3)
        representative_axis.legend()
        representative_figure.tight_layout()

        if save_dir is not None:
            representative_figure_path = (
                save_dir
                / (
                    f"{safe_subsystem_name}_"
                    f"representative_behaviour_{stage}h.png"
                )
            )

            representative_figure.savefig(
                representative_figure_path,
                dpi=300,
                bbox_inches="tight",
            )

            figure_paths[
                f"representative_behaviour_{stage}h"
            ] = representative_figure_path

        if show_plots:
            plt.show()
        else:
            plt.close(representative_figure)

    # ------------------------------------------------------
    # 14. Save presentation-ready tables
    # ------------------------------------------------------
    if save_dir is not None:
        central_summary_path = (
            save_dir
            / (
                f"{safe_subsystem_name}_"
                "stage_central_summary.csv"
            )
        )

        variability_summary_path = (
            save_dir
            / (
                f"{safe_subsystem_name}_"
                "stage_variability_summary.csv"
            )
        )

        representative_data_path = (
            save_dir
            / (
                f"{safe_subsystem_name}_"
                "representative_operating_data.csv"
            )
        )

        stage_central_summary.to_csv(
            central_summary_path,
            index=False,
        )

        stage_variability_summary.to_csv(
            variability_summary_path,
            index=False,
        )

        representative_data.to_csv(
            representative_data_path,
            index=False,
        )

        table_paths[
            "stage_central_summary"
        ] = central_summary_path

        table_paths[
            "stage_variability_summary"
        ] = variability_summary_path

        table_paths[
            "representative_operating_data"
        ] = representative_data_path

    # ------------------------------------------------------
    # 15. Return complete analysis outputs
    # ------------------------------------------------------
    return {
        # Presentation-ready tables
        "stage_central_summary": stage_central_summary,
        "stage_variability_summary": stage_variability_summary,

        # Raw analytical tables
        "stage_central_summary_raw":
            stage_central_summary_raw,
        "stage_variability_summary_raw":
            stage_variability_summary_raw,

        # Supporting data
        "representative_data": representative_data,
        "normalised_data": subsystem_df,
        "normalised_columns": normalised_columns,
        "constant_variables": constant_variables,
        "scaler": scaler,

        # Saved-output paths
        "figure_paths": figure_paths,
        "table_paths": table_paths,
    }
#-----------------------------------------------------------------------------------------------------
# =============================================================================
# COMPARATIVE SUBSYSTEM BEHAVIOUR ANALYSIS
# =============================================================================


# =============================================================================
# Internal helper functions
# =============================================================================

def _normalise_name(name: str) -> str:
    """
    Convert a column name into a simplified form for flexible matching.
    """
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> str | None:
    """
    Find a dataframe column using flexible case-insensitive matching.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe whose columns will be searched.
    candidates : sequence of str
        Possible column names.
    required : bool, default=True
        Raise an error when no matching column is found.

    Returns
    -------
    str or None
        Matching dataframe column.
    """
    exact_lookup = {
        _normalise_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalised_candidate = _normalise_name(candidate)

        if normalised_candidate in exact_lookup:
            return exact_lookup[normalised_candidate]

    if required:
        raise KeyError(
            "None of the expected columns were found. "
            f"Expected one of {list(candidates)}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def _ensure_directory(directory: str | Path) -> Path:
    """
    Create an output directory when it does not already exist.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_dataframe(
    dataframe: pd.DataFrame,
    path: str | Path,
) -> Path:
    """
    Save a dataframe as CSV without its index.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def _safe_min_max(series: pd.Series) -> pd.Series:
    """
    Apply Min-Max normalisation safely.

    A constant series is returned as zeros.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")
    minimum = numeric_series.min()
    maximum = numeric_series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(np.nan, index=series.index, dtype=float)

    data_range = maximum - minimum

    if np.isclose(data_range, 0):
        return pd.Series(0.0, index=series.index, dtype=float)

    return (numeric_series - minimum) / data_range


def _coerce_stage_values(series: pd.Series) -> pd.Series:
    """
    Convert stage labels such as '50 h' into numeric operating-hour values.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        return numeric.astype(float)

    extracted = (
        series.astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    )

    return pd.to_numeric(extracted, errors="coerce")


def _validate_subsystem_results(
    subsystem_name: str,
    results: Mapping[str, Any],
) -> None:
    """
    Validate that a subsystem result dictionary contains the required outputs.
    """
    required_keys = {
        "stage_central_summary_raw",
        "stage_variability_summary_raw",
    }

    missing_keys = required_keys.difference(results.keys())

    if missing_keys:
        raise KeyError(
            f"{subsystem_name} results are missing the following keys: "
            f"{sorted(missing_keys)}"
        )

    for key in required_keys:
        if not isinstance(results[key], pd.DataFrame):
            raise TypeError(
                f"{subsystem_name}['{key}'] must be a pandas DataFrame."
            )

        if results[key].empty:
            raise ValueError(
                f"{subsystem_name}['{key}'] is empty."
            )


def _get_stage_central_data(
    results: Mapping[str, Any],
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Extract stage-level normalised central behaviour.

    Returns
    -------
    dataframe : pandas.DataFrame
        Stage-level central summary.
    stage_column : str
        Name of the operating-stage column.
    value_columns : list of str
        Normalised variable columns used in the subsystem analysis.
    """
    dataframe = results["stage_central_summary_raw"].copy()

    stage_column = _find_column(
        dataframe,
        candidates=[
            "operating_hour",
            "operating hours",
            "operating stage",
            "stage",
            "hour",
            "hours",
            "test_hour",
            "durability_stage",
        ],
    )

    dataframe[stage_column] = _coerce_stage_values(dataframe[stage_column])
    dataframe = dataframe.dropna(subset=[stage_column])
    dataframe = dataframe.sort_values(stage_column).reset_index(drop=True)

    result_normalised_columns = list(results.get("normalised_columns", []))

    value_columns = [
        column
        for column in result_normalised_columns
        if column in dataframe.columns
    ]

    if not value_columns:
        numeric_columns = dataframe.select_dtypes(include=np.number).columns.tolist()

        value_columns = [
            column
            for column in numeric_columns
            if column != stage_column
            and (
                "normal" in str(column).lower()
                or str(column).lower().endswith("_scaled")
            )
        ]

    if not value_columns:
        numeric_columns = dataframe.select_dtypes(include=np.number).columns.tolist()
        value_columns = [
            column
            for column in numeric_columns
            if column != stage_column
        ]

    if not value_columns:
        raise ValueError(
            "No numeric stage-level behaviour variables were found."
        )

    dataframe[value_columns] = dataframe[value_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return dataframe, stage_column, value_columns


def _get_variability_data(
    results: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Extract and standardise the stage-level variability summary.

    Returns
    -------
    dataframe : pandas.DataFrame
        Long-format variability summary.
    column_map : dict
        Mapping from standard metric names to actual dataframe columns.
    """
    dataframe = results["stage_variability_summary_raw"].copy()

    column_map = {
        "stage": _find_column(
            dataframe,
            candidates=[
                "operating_hour",
                "operating hours",
                "operating stage",
                "stage",
                "hour",
                "hours",
                "durability_stage",
            ],
        ),
        "variable": _find_column(
            dataframe,
            candidates=[
                "variable",
                "variable_name",
                "feature",
                "parameter",
                "signal",
            ],
        ),
        "median": _find_column(
            dataframe,
            candidates=["median", "stage_median"],
            required=False,
        ),
        "q1": _find_column(
            dataframe,
            candidates=["q1", "first_quartile", "25%", "25th_percentile"],
            required=False,
        ),
        "q3": _find_column(
            dataframe,
            candidates=["q3", "third_quartile", "75%", "75th_percentile"],
            required=False,
        ),
        "iqr": _find_column(
            dataframe,
            candidates=[
                "iqr",
                "interquartile_range",
                "interquartile range",
            ],
            required=False,
        ),
        "std": _find_column(
            dataframe,
            candidates=[
                "std",
                "standard_deviation",
                "standard deviation",
                "sd",
            ],
            required=False,
        ),
    }

    dataframe[column_map["stage"]] = _coerce_stage_values(
        dataframe[column_map["stage"]]
    )

    numeric_metric_columns = [
        column
        for key, column in column_map.items()
        if key not in {"stage", "variable"} and column is not None
    ]

    dataframe[numeric_metric_columns] = dataframe[numeric_metric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if column_map["iqr"] is None:
        if column_map["q1"] is None or column_map["q3"] is None:
            raise KeyError(
                "The variability summary must contain either an IQR column "
                "or both Q1 and Q3 columns."
            )

        dataframe["_calculated_iqr"] = (
            dataframe[column_map["q3"]]
            - dataframe[column_map["q1"]]
        )
        column_map["iqr"] = "_calculated_iqr"

    if column_map["std"] is None:
        raise KeyError(
            "The variability summary must contain a standard deviation column."
        )

    dataframe = dataframe.dropna(subset=[column_map["stage"]])
    dataframe = dataframe.sort_values(
        [column_map["stage"], column_map["variable"]]
    ).reset_index(drop=True)

    return dataframe, column_map


def _calculate_synchronisation_score(
    stage_data: pd.DataFrame,
    value_columns: Sequence[str],
) -> float:
    """
    Calculate descriptive within-subsystem behavioural synchronisation.

    Synchronisation is represented by the median absolute Spearman correlation
    between stage-level variable trajectories.

    Returns
    -------
    float
        Value between zero and one, where a higher value indicates more
        similar stage-level evolution among variables.
    """
    usable_columns = [
        column
        for column in value_columns
        if stage_data[column].nunique(dropna=True) > 1
    ]

    if len(usable_columns) < 2:
        return np.nan

    correlation_matrix = stage_data[usable_columns].corr(
        method="spearman"
    ).abs()

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape, dtype=bool),
            k=1,
        )
    )

    pairwise_correlations = upper_triangle.stack()

    if pairwise_correlations.empty:
        return np.nan

    return float(pairwise_correlations.median())


def _calculate_stage_behaviour_profile(
    subsystem_name: str,
    results: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Create a stage-level subsystem behaviour profile.

    The profile includes:

    - mean normalised subsystem behaviour;
    - mean absolute displacement from the first stage;
    - stage-to-stage transition magnitude;
    - median IQR;
    - median standard deviation.
    """
    central_data, stage_column, value_columns = _get_stage_central_data(results)
    variability_data, variability_columns = _get_variability_data(results)

    behaviour_matrix = central_data[value_columns]

    baseline = behaviour_matrix.iloc[0]

    profile = pd.DataFrame(
        {
            "subsystem": subsystem_name,
            "operating_hour": central_data[stage_column].to_numpy(),
            "mean_normalised_behaviour": behaviour_matrix.mean(
                axis=1,
                skipna=True,
            ).to_numpy(),
            "mean_absolute_displacement": behaviour_matrix.sub(
                baseline,
                axis="columns",
            ).abs().mean(
                axis=1,
                skipna=True,
            ).to_numpy(),
            "maximum_absolute_displacement": behaviour_matrix.sub(
                baseline,
                axis="columns",
            ).abs().max(
                axis=1,
                skipna=True,
            ).to_numpy(),
        }
    )

    profile["transition_magnitude"] = (
        behaviour_matrix.diff()
        .abs()
        .mean(axis=1, skipna=True)
        .to_numpy()
    )

    profile.loc[profile.index[0], "transition_magnitude"] = 0.0

    stage_variability = (
        variability_data.groupby(variability_columns["stage"], as_index=False)
        .agg(
            median_iqr=(variability_columns["iqr"], "median"),
            mean_iqr=(variability_columns["iqr"], "mean"),
            median_standard_deviation=(variability_columns["std"], "median"),
            mean_standard_deviation=(variability_columns["std"], "mean"),
        )
        .rename(columns={variability_columns["stage"]: "operating_hour"})
    )

    profile = profile.merge(
        stage_variability,
        on="operating_hour",
        how="left",
    )

    return profile


def _extract_representative_windows(
    results: Mapping[str, Any],
) -> dict[float, pd.DataFrame]:
    """
    Extract representative operating windows from a subsystem result.

    Supported structures
    --------------------
    1. Dictionary:
       {50: dataframe, 550: dataframe, 1000: dataframe}

    2. Single dataframe containing a stage or operating-hour column.

    Returns
    -------
    dict
        Dictionary mapping operating stage to representative dataframe.
    """
    representative_data = results.get("representative_data")

    if representative_data is None:
        return {}

    windows: dict[float, pd.DataFrame] = {}

    if isinstance(representative_data, Mapping):
        for stage, dataframe in representative_data.items():
            if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
                continue

            stage_value = pd.to_numeric(
                pd.Series([stage]).astype(str).str.extract(
                    r"([-+]?\d*\.?\d+)",
                    expand=False,
                ),
                errors="coerce",
            ).iloc[0]

            if pd.notna(stage_value):
                windows[float(stage_value)] = dataframe.copy()

        return dict(sorted(windows.items()))

    if isinstance(representative_data, pd.DataFrame):
        dataframe = representative_data.copy()

        stage_column = _find_column(
            dataframe,
            candidates=[
                "operating_hour",
                "operating hours",
                "operating stage",
                "representative_stage",
                "stage",
                "hour",
                "hours",
            ],
            required=False,
        )

        if stage_column is None:
            return {}

        dataframe["_comparative_stage"] = _coerce_stage_values(
            dataframe[stage_column]
        )

        for stage, stage_dataframe in dataframe.groupby("_comparative_stage"):
            if pd.notna(stage):
                windows[float(stage)] = (
                    stage_dataframe
                    .drop(columns="_comparative_stage")
                    .reset_index(drop=True)
                )

        return dict(sorted(windows.items()))

    return {}


def _select_representative_value_columns(
    dataframe: pd.DataFrame,
    results: Mapping[str, Any],
) -> list[str]:
    """
    Select normalised variables from a representative operating dataframe.
    """
    result_columns = list(results.get("normalised_columns", []))

    selected_columns = [
        column
        for column in result_columns
        if column in dataframe.columns
    ]

    if selected_columns:
        return selected_columns

    numeric_columns = dataframe.select_dtypes(include=np.number).columns.tolist()

    excluded_names = {
        "time",
        "relative_time",
        "elapsed_time",
        "operating_hour",
        "stage",
        "hour",
        "index",
    }

    selected_columns = [
        column
        for column in numeric_columns
        if _normalise_name(column) not in {
            _normalise_name(name) for name in excluded_names
        }
        and (
            "normal" in str(column).lower()
            or str(column).lower().endswith("_scaled")
        )
    ]

    if not selected_columns:
        selected_columns = [
            column
            for column in numeric_columns
            if _normalise_name(column) not in {
                _normalise_name(name) for name in excluded_names
            }
        ]

    return selected_columns


def _create_resampled_signature(
    dataframe: pd.DataFrame,
    value_columns: Sequence[str],
    number_of_points: int = 200,
) -> pd.DataFrame:
    """
    Create a fixed-length subsystem operating signature.

    Each variable is independently Min-Max normalised within the representative
    window. Variables are then averaged to produce a subsystem-level operating
    signature.
    """
    if number_of_points < 2:
        raise ValueError("number_of_points must be at least 2.")

    data = dataframe[list(value_columns)].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.interpolate(limit_direction="both")
    data = data.dropna(how="all")

    if len(data) < 2:
        return pd.DataFrame(
            columns=["relative_position", "composite_signature"]
        )

    normalised_data = data.apply(_safe_min_max)

    composite_signature = normalised_data.mean(axis=1, skipna=True)

    source_position = np.linspace(0.0, 1.0, len(composite_signature))
    target_position = np.linspace(0.0, 1.0, number_of_points)

    valid_mask = composite_signature.notna().to_numpy()

    if valid_mask.sum() < 2:
        return pd.DataFrame(
            columns=["relative_position", "composite_signature"]
        )

    interpolated_signature = np.interp(
        target_position,
        source_position[valid_mask],
        composite_signature.to_numpy()[valid_mask],
    )

    return pd.DataFrame(
        {
            "relative_position": target_position,
            "composite_signature": interpolated_signature,
        }
    )


def _calculate_representative_repeatability(
    results: Mapping[str, Any],
    number_of_points: int = 200,
) -> float:
    """
    Calculate repeatability across representative operating stages.

    Repeatability is measured using the median pairwise correlation between
    fixed-length subsystem operating signatures.

    Returns
    -------
    float
        Correlation-based repeatability score between zero and one.
    """
    windows = _extract_representative_windows(results)

    signatures: dict[float, pd.Series] = {}

    for stage, dataframe in windows.items():
        value_columns = _select_representative_value_columns(
            dataframe,
            results,
        )

        if not value_columns:
            continue

        signature = _create_resampled_signature(
            dataframe,
            value_columns=value_columns,
            number_of_points=number_of_points,
        )

        if not signature.empty:
            signatures[stage] = signature["composite_signature"]

    if len(signatures) < 2:
        return np.nan

    signature_dataframe = pd.DataFrame(signatures)
    correlations = signature_dataframe.corr().abs()

    upper_triangle = correlations.where(
        np.triu(
            np.ones(correlations.shape, dtype=bool),
            k=1,
        )
    )

    pairwise_correlations = upper_triangle.stack()

    if pairwise_correlations.empty:
        return np.nan

    return float(pairwise_correlations.median())


def _classify_relative_level(
    values: pd.Series,
    higher_is_more: bool = True,
) -> pd.Series:
    """
    Classify a metric into relative Low, Moderate, or High categories.

    Classification is relative to the four subsystems and should not be
    interpreted as an absolute engineering threshold.
    """
    numeric_values = pd.to_numeric(values, errors="coerce")

    if numeric_values.notna().sum() < 2:
        return pd.Series(
            ["Not available"] * len(values),
            index=values.index,
            dtype="object",
        )

    if np.isclose(
        numeric_values.max(skipna=True),
        numeric_values.min(skipna=True),
    ):
        return pd.Series(
            ["Similar"] * len(values),
            index=values.index,
            dtype="object",
        )

    percentile_rank = numeric_values.rank(
        method="average",
        pct=True,
        na_option="keep",
    )

    if not higher_is_more:
        percentile_rank = 1.0 - percentile_rank + (
            1.0 / numeric_values.notna().sum()
        )

    labels = pd.Series(
        "Moderate",
        index=values.index,
        dtype="object",
    )

    labels.loc[percentile_rank <= 0.34] = "Low"
    labels.loc[percentile_rank >= 0.67] = "High"
    labels.loc[numeric_values.isna()] = "Not available"

    return labels


# =============================================================================
# Metric generation
# =============================================================================

def calculate_subsystem_metrics(
    subsystem_name: str,
    results: Mapping[str, Any],
    representative_points: int = 200,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calculate descriptive subsystem-level behavioural metrics.

    Parameters
    ----------
    subsystem_name : str
        Name used in tables and figures.
    results : mapping
        Output dictionary returned by analyse_subsystem_behaviour().
    representative_points : int, default=200
        Number of points used when comparing representative signatures.

    Returns
    -------
    subsystem_metrics : pandas.Series
        Overall subsystem-level behavioural metrics.
    stage_profile : pandas.DataFrame
        Stage-level comparative behaviour profile.

    Notes
    -----
    All metrics are descriptive. They compare relative behaviour and do not
    establish statistical dependence or causal relationships.
    """
    _validate_subsystem_results(subsystem_name, results)

    stage_profile = _calculate_stage_behaviour_profile(
        subsystem_name,
        results,
    )

    central_data, _, value_columns = _get_stage_central_data(results)

    synchronisation_score = _calculate_synchronisation_score(
        central_data,
        value_columns,
    )

    repeatability_score = _calculate_representative_repeatability(
        results,
        number_of_points=representative_points,
    )

    initial_mean_behaviour = stage_profile[
        "mean_normalised_behaviour"
    ].iloc[0]

    final_mean_behaviour = stage_profile[
        "mean_normalised_behaviour"
    ].iloc[-1]

    overall_signed_shift = (
        final_mean_behaviour - initial_mean_behaviour
    )

    metrics = pd.Series(
        {
            "subsystem": subsystem_name,
            "number_of_variables": len(value_columns),
            "number_of_stages": stage_profile["operating_hour"].nunique(),
            "initial_mean_normalised_behaviour": initial_mean_behaviour,
            "final_mean_normalised_behaviour": final_mean_behaviour,
            "overall_signed_shift": overall_signed_shift,
            "final_absolute_displacement": stage_profile[
                "mean_absolute_displacement"
            ].iloc[-1],
            "maximum_absolute_displacement": stage_profile[
                "mean_absolute_displacement"
            ].max(),
            "median_transition_magnitude": stage_profile[
                "transition_magnitude"
            ].median(),
            "maximum_transition_magnitude": stage_profile[
                "transition_magnitude"
            ].max(),
            "median_iqr": stage_profile["median_iqr"].median(),
            "mean_iqr": stage_profile["mean_iqr"].mean(),
            "median_standard_deviation": stage_profile[
                "median_standard_deviation"
            ].median(),
            "mean_standard_deviation": stage_profile[
                "mean_standard_deviation"
            ].mean(),
            "synchronisation_score": synchronisation_score,
            "representative_repeatability_score": repeatability_score,
        }
    )

    return metrics, stage_profile


def create_comparison_summary(
    subsystem_results: Mapping[str, Mapping[str, Any]],
    representative_points: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create overall and stage-level comparative behaviour summaries.

    Parameters
    ----------
    subsystem_results : mapping
        Dictionary in the following form:

        {
            "Electrical": electrical_results,
            "Pressure": pressure_results,
            "Temperature": temperature_results,
            "Reactant Flow": reactant_flow_results,
        }

    representative_points : int, default=200
        Number of points used to construct representative signatures.

    Returns
    -------
    comparison_summary : pandas.DataFrame
        One row per engineering subsystem.
    stage_profiles : pandas.DataFrame
        Stage-level behaviour profiles for all subsystems.
    """
    if not isinstance(subsystem_results, Mapping):
        raise TypeError("subsystem_results must be a mapping.")

    if len(subsystem_results) < 2:
        raise ValueError(
            "At least two subsystems are required for comparison."
        )

    metric_rows: list[pd.Series] = []
    profile_frames: list[pd.DataFrame] = []

    for subsystem_name, results in subsystem_results.items():
        metrics, profile = calculate_subsystem_metrics(
            subsystem_name=subsystem_name,
            results=results,
            representative_points=representative_points,
        )

        metric_rows.append(metrics)
        profile_frames.append(profile)

    comparison_summary = pd.DataFrame(metric_rows).reset_index(drop=True)

    stage_profiles = pd.concat(
        profile_frames,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Relative descriptive indices
    # -------------------------------------------------------------------------
    variability_components = comparison_summary[
        ["median_iqr", "median_standard_deviation"]
    ].copy()

    for column in variability_components.columns:
        variability_components[column] = _safe_min_max(
            variability_components[column]
        )

    comparison_summary["relative_variability_index"] = (
        variability_components.mean(axis=1, skipna=True)
    )

    comparison_summary["relative_stability_score"] = (
        1.0 - comparison_summary["relative_variability_index"]
    )

    transition_components = comparison_summary[
        [
            "median_transition_magnitude",
            "maximum_transition_magnitude",
        ]
    ].copy()

    for column in transition_components.columns:
        transition_components[column] = _safe_min_max(
            transition_components[column]
        )

    comparison_summary["relative_transition_index"] = (
        transition_components.mean(axis=1, skipna=True)
    )

    evolution_components = comparison_summary[
        [
            "final_absolute_displacement",
            "maximum_absolute_displacement",
        ]
    ].copy()

    for column in evolution_components.columns:
        evolution_components[column] = _safe_min_max(
            evolution_components[column]
        )

    comparison_summary["relative_evolution_index"] = (
        evolution_components.mean(axis=1, skipna=True)
    )

    # -------------------------------------------------------------------------
    # Relative qualitative descriptions
    # -------------------------------------------------------------------------
    comparison_summary["relative_stability"] = _classify_relative_level(
        comparison_summary["relative_stability_score"],
        higher_is_more=True,
    )

    comparison_summary["relative_variability"] = _classify_relative_level(
        comparison_summary["relative_variability_index"],
        higher_is_more=True,
    )

    comparison_summary["relative_transition_activity"] = (
        _classify_relative_level(
            comparison_summary["relative_transition_index"],
            higher_is_more=True,
        )
    )

    comparison_summary["relative_evolution"] = _classify_relative_level(
        comparison_summary["relative_evolution_index"],
        higher_is_more=True,
    )

    comparison_summary["within_subsystem_synchronisation"] = (
        _classify_relative_level(
            comparison_summary["synchronisation_score"],
            higher_is_more=True,
        )
    )

    comparison_summary["operating_repeatability"] = (
        _classify_relative_level(
            comparison_summary["representative_repeatability_score"],
            higher_is_more=True,
        )
    )

    ordered_columns = [
        "subsystem",
        "number_of_variables",
        "number_of_stages",
        "initial_mean_normalised_behaviour",
        "final_mean_normalised_behaviour",
        "overall_signed_shift",
        "final_absolute_displacement",
        "maximum_absolute_displacement",
        "median_transition_magnitude",
        "maximum_transition_magnitude",
        "median_iqr",
        "mean_iqr",
        "median_standard_deviation",
        "mean_standard_deviation",
        "relative_evolution_index",
        "relative_transition_index",
        "relative_variability_index",
        "relative_stability_score",
        "synchronisation_score",
        "representative_repeatability_score",
        "relative_evolution",
        "relative_transition_activity",
        "relative_stability",
        "relative_variability",
        "within_subsystem_synchronisation",
        "operating_repeatability",
    ]

    comparison_summary = comparison_summary[ordered_columns]

    return comparison_summary, stage_profiles


def create_behaviour_comparison_table(
    comparison_summary: pd.DataFrame,
    decimal_places: int = 4,
) -> pd.DataFrame:
    """
    Create a concise, presentation-ready comparative behaviour table.
    """
    required_columns = {
        "subsystem",
        "relative_evolution",
        "relative_transition_activity",
        "relative_stability",
        "relative_variability",
        "within_subsystem_synchronisation",
        "operating_repeatability",
        "relative_evolution_index",
        "relative_transition_index",
        "relative_stability_score",
        "relative_variability_index",
        "synchronisation_score",
        "representative_repeatability_score",
    }

    missing_columns = required_columns.difference(
        comparison_summary.columns
    )

    if missing_columns:
        raise KeyError(
            "comparison_summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    table = comparison_summary[
        [
            "subsystem",
            "relative_evolution",
            "relative_transition_activity",
            "relative_stability",
            "relative_variability",
            "within_subsystem_synchronisation",
            "operating_repeatability",
            "relative_evolution_index",
            "relative_transition_index",
            "relative_stability_score",
            "relative_variability_index",
            "synchronisation_score",
            "representative_repeatability_score",
        ]
    ].copy()

    table = table.rename(
        columns={
            "subsystem": "Subsystem",
            "relative_evolution": "Behavioural Evolution",
            "relative_transition_activity": "Transition Activity",
            "relative_stability": "Relative Stability",
            "relative_variability": "Relative Variability",
            "within_subsystem_synchronisation": "Synchronisation",
            "operating_repeatability": "Operating Repeatability",
            "relative_evolution_index": "Evolution Index",
            "relative_transition_index": "Transition Index",
            "relative_stability_score": "Stability Score",
            "relative_variability_index": "Variability Index",
            "synchronisation_score": "Synchronisation Score",
            "representative_repeatability_score": "Repeatability Score",
        }
    )

    numeric_columns = table.select_dtypes(include=np.number).columns
    table[numeric_columns] = table[numeric_columns].round(decimal_places)

    return table


def create_pairwise_comparison_table(
    comparison_summary: pd.DataFrame,
    decimal_places: int = 4,
) -> pd.DataFrame:
    """
    Create all pairwise subsystem comparisons.

    The table compares the relative metric differences between each pair of
    subsystems. Positive values indicate that subsystem A has the larger
    metric; negative values indicate that subsystem B has the larger metric.
    """
    required_columns = {
        "subsystem",
        "relative_stability_score",
        "relative_variability_index",
        "relative_transition_index",
        "relative_evolution_index",
        "synchronisation_score",
        "representative_repeatability_score",
    }

    missing_columns = required_columns.difference(
        comparison_summary.columns
    )

    if missing_columns:
        raise KeyError(
            "comparison_summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    records: list[dict[str, Any]] = []

    for first_index in range(len(comparison_summary)):
        for second_index in range(first_index + 1, len(comparison_summary)):
            subsystem_a = comparison_summary.iloc[first_index]
            subsystem_b = comparison_summary.iloc[second_index]

            records.append(
                {
                    "Subsystem A": subsystem_a["subsystem"],
                    "Subsystem B": subsystem_b["subsystem"],
                    "Stability Difference": (
                        subsystem_a["relative_stability_score"]
                        - subsystem_b["relative_stability_score"]
                    ),
                    "Variability Difference": (
                        subsystem_a["relative_variability_index"]
                        - subsystem_b["relative_variability_index"]
                    ),
                    "Transition Difference": (
                        subsystem_a["relative_transition_index"]
                        - subsystem_b["relative_transition_index"]
                    ),
                    "Evolution Difference": (
                        subsystem_a["relative_evolution_index"]
                        - subsystem_b["relative_evolution_index"]
                    ),
                    "Synchronisation Difference": (
                        subsystem_a["synchronisation_score"]
                        - subsystem_b["synchronisation_score"]
                    ),
                    "Repeatability Difference": (
                        subsystem_a["representative_repeatability_score"]
                        - subsystem_b["representative_repeatability_score"]
                    ),
                }
            )

    pairwise_table = pd.DataFrame(records)

    numeric_columns = pairwise_table.select_dtypes(include=np.number).columns
    pairwise_table[numeric_columns] = pairwise_table[numeric_columns].round(
        decimal_places
    )

    return pairwise_table


# =============================================================================
# Comparative figures
# =============================================================================

def plot_subsystem_behaviour_evolution(
    stage_profiles: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "subsystem_behaviour_evolution_comparison.png",
    dpi: int = 300,
    show: bool = True,
) -> Path:
    """
    Plot mean absolute subsystem displacement from the first stage.
    """
    required_columns = {
        "subsystem",
        "operating_hour",
        "mean_absolute_displacement",
    }

    missing_columns = required_columns.difference(stage_profiles.columns)

    if missing_columns:
        raise KeyError(
            f"stage_profiles is missing columns: {sorted(missing_columns)}"
        )

    output_directory = _ensure_directory(output_dir)
    output_path = output_directory / filename

    figure, axis = plt.subplots(figsize=(11, 6))

    for subsystem, data in stage_profiles.groupby("subsystem"):
        ordered_data = data.sort_values("operating_hour")

        axis.plot(
            ordered_data["operating_hour"],
            ordered_data["mean_absolute_displacement"],
            marker="o",
            linewidth=2,
            label=subsystem,
        )

    axis.set_title("Subsystem Behaviour Evolution Comparison")
    axis.set_xlabel("Operating Hour")
    axis.set_ylabel("Mean Absolute Displacement from Initial Stage")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Subsystem")
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(figure)

    return output_path


def plot_subsystem_stability_comparison(
    comparison_summary: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "subsystem_stability_comparison.png",
    dpi: int = 300,
    show: bool = True,
) -> Path:
    """
    Plot the relative stability score for each subsystem.

    The score is comparative:
        1 - relative variability index

    It is not an absolute engineering stability threshold.
    """
    required_columns = {
        "subsystem",
        "relative_stability_score",
    }

    missing_columns = required_columns.difference(
        comparison_summary.columns
    )

    if missing_columns:
        raise KeyError(
            f"comparison_summary is missing columns: {sorted(missing_columns)}"
        )

    plot_data = comparison_summary.sort_values(
        "relative_stability_score",
        ascending=False,
    )

    output_directory = _ensure_directory(output_dir)
    output_path = output_directory / filename

    figure, axis = plt.subplots(figsize=(9, 6))

    bars = axis.bar(
        plot_data["subsystem"],
        plot_data["relative_stability_score"],
    )

    axis.set_title("Relative Subsystem Stability Comparison")
    axis.set_xlabel("Engineering Subsystem")
    axis.set_ylabel("Relative Stability Score")
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", alpha=0.3)

    for bar, value in zip(
        bars,
        plot_data["relative_stability_score"],
    ):
        if pd.notna(value):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(figure)

    return output_path


def plot_subsystem_variability_comparison(
    comparison_summary: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "subsystem_variability_comparison.png",
    dpi: int = 300,
    show: bool = True,
) -> Path:
    """
    Plot median normalised IQR and standard deviation for each subsystem.
    """
    required_columns = {
        "subsystem",
        "median_iqr",
        "median_standard_deviation",
    }

    missing_columns = required_columns.difference(
        comparison_summary.columns
    )

    if missing_columns:
        raise KeyError(
            f"comparison_summary is missing columns: {sorted(missing_columns)}"
        )

    plot_data = comparison_summary.copy()

    output_directory = _ensure_directory(output_dir)
    output_path = output_directory / filename

    x_positions = np.arange(len(plot_data))
    bar_width = 0.36

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.bar(
        x_positions - bar_width / 2,
        plot_data["median_iqr"],
        width=bar_width,
        label="Median IQR",
    )

    axis.bar(
        x_positions + bar_width / 2,
        plot_data["median_standard_deviation"],
        width=bar_width,
        label="Median Standard Deviation",
    )

    axis.set_title("Subsystem Variability Comparison")
    axis.set_xlabel("Engineering Subsystem")
    axis.set_ylabel("Normalised Variability")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(plot_data["subsystem"])
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(figure)

    return output_path


def plot_subsystem_transition_comparison(
    stage_profiles: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "subsystem_transition_comparison.png",
    dpi: int = 300,
    show: bool = True,
) -> Path:
    """
    Plot stage-to-stage transition magnitude for every subsystem.
    """
    required_columns = {
        "subsystem",
        "operating_hour",
        "transition_magnitude",
    }

    missing_columns = required_columns.difference(stage_profiles.columns)

    if missing_columns:
        raise KeyError(
            f"stage_profiles is missing columns: {sorted(missing_columns)}"
        )

    output_directory = _ensure_directory(output_dir)
    output_path = output_directory / filename

    figure, axis = plt.subplots(figsize=(11, 6))

    for subsystem, data in stage_profiles.groupby("subsystem"):
        ordered_data = data.sort_values("operating_hour")

        axis.plot(
            ordered_data["operating_hour"],
            ordered_data["transition_magnitude"],
            marker="o",
            linewidth=2,
            label=subsystem,
        )

    axis.set_title("Subsystem Behaviour Transition Comparison")
    axis.set_xlabel("Operating Hour")
    axis.set_ylabel("Mean Stage-to-Stage Transition Magnitude")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Subsystem")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(figure)

    return output_path


def plot_representative_operating_comparison(
    subsystem_results: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    representative_stages: Sequence[float] | None = None,
    number_of_points: int = 200,
    filename: str = "representative_operating_behaviour_comparison.png",
    dpi: int = 300,
    show: bool = True,
) -> Path | None:
    """
    Compare subsystem-level representative operating signatures.

    A separate axis is created for each subsystem. Within each axis, the
    selected early, middle, and late operating stages are compared.

    Returns
    -------
    pathlib.Path or None
        Figure path, or None when representative data are unavailable.
    """
    output_directory = _ensure_directory(output_dir)
    output_path = output_directory / filename

    available_subsystems: dict[
        str,
        dict[float, pd.DataFrame],
    ] = {}

    for subsystem_name, results in subsystem_results.items():
        windows = _extract_representative_windows(results)

        if windows:
            available_subsystems[subsystem_name] = windows

    if not available_subsystems:
        warnings.warn(
            "Representative operating data were not available. "
            "The representative comparison figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    number_of_subsystems = len(available_subsystems)

    figure, axes = plt.subplots(
        nrows=number_of_subsystems,
        ncols=1,
        figsize=(12, 4 * number_of_subsystems),
        squeeze=False,
    )

    for axis, (subsystem_name, windows) in zip(
        axes.flatten(),
        available_subsystems.items(),
    ):
        available_stages = sorted(windows)

        if representative_stages is None:
            selected_stages = available_stages
        else:
            selected_stages = [
                min(
                    available_stages,
                    key=lambda stage: abs(stage - requested_stage),
                )
                for requested_stage in representative_stages
            ]

            selected_stages = list(dict.fromkeys(selected_stages))

        results = subsystem_results[subsystem_name]

        for stage in selected_stages:
            dataframe = windows[stage]

            value_columns = _select_representative_value_columns(
                dataframe,
                results,
            )

            if not value_columns:
                continue

            signature = _create_resampled_signature(
                dataframe=dataframe,
                value_columns=value_columns,
                number_of_points=number_of_points,
            )

            if signature.empty:
                continue

            axis.plot(
                signature["relative_position"],
                signature["composite_signature"],
                linewidth=1.8,
                label=f"{stage:g} h",
            )

        axis.set_title(f"{subsystem_name} Representative Operating Behaviour")
        axis.set_xlabel("Relative Position within Operating Window")
        axis.set_ylabel("Composite Normalised Behaviour")
        axis.grid(True, alpha=0.3)
        axis.legend(title="Operating Stage")

    figure.suptitle(
        "Representative Operating Behaviour Comparison",
        fontsize=14,
        y=1.01,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(figure)

    return output_path


# =============================================================================
# Main reusable wrapper function
# =============================================================================

def compare_subsystem_behaviour(
    electrical_results: Mapping[str, Any],
    pressure_results: Mapping[str, Any],
    temperature_results: Mapping[str, Any],
    reactant_flow_results: Mapping[str, Any],
    figures_dir: str | Path,
    results_dir: str | Path,
    representative_stages: Sequence[float] = (50, 550, 1000),
    representative_points: int = 200,
    decimal_places: int = 4,
    dpi: int = 300,
    show_figures: bool = True,
) -> dict[str, Any]:
    """
    Perform comparative behaviour analysis across the four engineering
    subsystems.

    Parameters
    ----------
    electrical_results : mapping
        Results returned by analyse_subsystem_behaviour() for the electrical
        subsystem.
    pressure_results : mapping
        Results returned for the pressure subsystem.
    temperature_results : mapping
        Results returned for the temperature subsystem.
    reactant_flow_results : mapping
        Results returned for the reactant flow subsystem.
    figures_dir : str or pathlib.Path
        Directory in which comparative figures will be saved.
    results_dir : str or pathlib.Path
        Directory in which comparative tables will be saved.
    representative_stages : sequence of float, default=(50, 550, 1000)
        Requested early, middle, and late operating stages.
    representative_points : int, default=200
        Number of points used to resample representative operating windows.
    decimal_places : int, default=4
        Number of decimal places in presentation-ready tables.
    dpi : int, default=300
        Resolution of saved figures.
    show_figures : bool, default=True
        Display figures in the notebook.

    Returns
    -------
    dict
        Dictionary containing:

        - comparison_summary
        - comparison_summary_raw
        - pairwise_comparison
        - stage_profiles
        - representative_signatures
        - figure_paths
        - table_paths

    Notes
    -----
    The analysis is descriptive and comparative. It identifies similarities
    and differences between engineering subsystems without inferring causal
    relationships.
    """
    subsystem_results = {
        "Electrical": electrical_results,
        "Pressure": pressure_results,
        "Temperature": temperature_results,
        "Reactant Flow": reactant_flow_results,
    }

    figures_output_dir = _ensure_directory(
        Path(figures_dir) / "comparative_behaviour"
    )

    tables_output_dir = _ensure_directory(
        Path(results_dir) / "comparative_behaviour"
    )

    # -------------------------------------------------------------------------
    # Generate comparative metrics
    # -------------------------------------------------------------------------
    comparison_summary_raw, stage_profiles = create_comparison_summary(
        subsystem_results=subsystem_results,
        representative_points=representative_points,
    )

    comparison_summary = create_behaviour_comparison_table(
        comparison_summary=comparison_summary_raw,
        decimal_places=decimal_places,
    )

    pairwise_comparison = create_pairwise_comparison_table(
        comparison_summary=comparison_summary_raw,
        decimal_places=decimal_places,
    )

    # -------------------------------------------------------------------------
    # Generate and save figures
    # -------------------------------------------------------------------------
    figure_paths: dict[str, Path | None] = {}

    figure_paths["behaviour_evolution"] = (
        plot_subsystem_behaviour_evolution(
            stage_profiles=stage_profiles,
            output_dir=figures_output_dir,
            dpi=dpi,
            show=show_figures,
        )
    )

    figure_paths["stability_comparison"] = (
        plot_subsystem_stability_comparison(
            comparison_summary=comparison_summary_raw,
            output_dir=figures_output_dir,
            dpi=dpi,
            show=show_figures,
        )
    )

    figure_paths["variability_comparison"] = (
        plot_subsystem_variability_comparison(
            comparison_summary=comparison_summary_raw,
            output_dir=figures_output_dir,
            dpi=dpi,
            show=show_figures,
        )
    )

    figure_paths["transition_comparison"] = (
        plot_subsystem_transition_comparison(
            stage_profiles=stage_profiles,
            output_dir=figures_output_dir,
            dpi=dpi,
            show=show_figures,
        )
    )

    figure_paths["representative_operating_comparison"] = (
        plot_representative_operating_comparison(
            subsystem_results=subsystem_results,
            output_dir=figures_output_dir,
            representative_stages=representative_stages,
            number_of_points=representative_points,
            dpi=dpi,
            show=show_figures,
        )
    )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------
    table_paths = {
        "comparison_summary": _save_dataframe(
            comparison_summary,
            tables_output_dir
            / "comparative_behaviour_summary.csv",
        ),
        "comparison_summary_raw": _save_dataframe(
            comparison_summary_raw,
            tables_output_dir
            / "comparative_behaviour_summary_raw.csv",
        ),
        "pairwise_comparison": _save_dataframe(
            pairwise_comparison,
            tables_output_dir
            / "pairwise_subsystem_comparison.csv",
        ),
        "stage_profiles": _save_dataframe(
            stage_profiles,
            tables_output_dir
            / "comparative_stage_profiles.csv",
        ),
    }

    return {
        "comparison_summary": comparison_summary,
        "comparison_summary_raw": comparison_summary_raw,
        "pairwise_comparison": pairwise_comparison,
        "stage_profiles": stage_profiles,
        "subsystem_results": subsystem_results,
        "figure_paths": figure_paths,
        "table_paths": table_paths,
    }