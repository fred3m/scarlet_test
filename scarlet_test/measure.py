import os
from typing import List, Sequence, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

from .core import __DATA_PATH__, get_filename, get_prs


def adjacent_values(vals: np.ndarray, q1: int, q3: int) -> Tuple[np.ndarray, np.ndarray]:
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def measure_blend(
        data: np.ndarray,
        sources: List,
        filters: Sequence[str],
) -> List[Dict[str, float]]:
    """
    Measure all of the fake sources in a single blend

    :param data: The numpy file with blend data
    :param sources: The sources in the blend
    :param filters: The filter name for each band
    :return: List of measurements for each matched source
    """
    import scarlet.measure

    # Extract necessary fields from the data
    centers = data["centers"]
    matched = data["matched"]
    matched_centers = np.array([[m["y"], m["x"]] for m in matched]).astype(int)

    true_flux = np.array([matched[f + "magVar"] for f in filters])

    measurements = []
    for k, (cy, cx) in enumerate(matched_centers):
        # Get the matching index for the source based on its center
        matched_idx = np.where((centers[:, 0] == cy) & (centers[:, 1] == cx))[0][0]

        # Calculate the flux difference in each band
        source = sources[matched_idx]
        flux = 27 - 2.5*np.log10(scarlet.measure.flux(source))

        diff = true_flux[:, k] - flux

        measurement = {filters[f] + " diff": diff[f] for f in range(len(filters))}
        measurements.append(measurement)

    return measurements


def check_log(data, ax):
    _data = np.log10(data)
    ymin, ymax = np.min(_data), np.max(_data)
    # Use a log scale if the range is more than 2 orders of magnitude
    if ymax - ymin > 2:
        ymin = int(np.max([1e-50, ymin - 1]))
        ymax = int(ymax+1)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ax.yaxis.set_ticks([
            np.log10(x) for p in range(ymin, ymax)
            for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
        return True
    return False


class Metric:
    """
    A metric to be calculated based on a set of deblended sources
    """
    def __init__(
            self,
            name: str,
            units: str,
            use_abs: bool = False
    ):
        """
        Initialize the class
        :param name: Name of the metric.
        :param units: Units of the metric.
        :param use_abs: Whether or not this metric is an absolute value
        """
        self.name = name
        self.units = units
        self.use_abs = use_abs

    def plot(
            self,
            set_id: str,
            scatter_prs: List[str] = None,
            plot_prs: List[str] = None,
            data_path: str = None,
    ) -> None:
        """
        Create a plot using the records for a given set ID.

        :param set_id: ID of the set to analyze
        :param scatter_prs: A list of pull requests to plot.
            If `prs` is `None` then only the last 2 PRs are plotted.
        :param plot_prs: A list of pull requests to plot.
            If `prs` is `None` then all PRs are shown.
        :parama data_path: Location of the measurement data.
        """
        if data_path is None:
            data_path = os.path.join(__DATA_PATH__, set_id)
        all_prs = get_prs()
        if scatter_prs is None:
            # Use the last two PRs
            scatter_prs = all_prs[-2:]
        if plot_prs is None:
            # Plot all the PRs if none are specified
            plot_prs = all_prs

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # First display all of the scatter plots for the two most recent PRs
        records = {
            pr: np.load(os.path.join(data_path, get_filename(pr)))["records"]
            for pr in scatter_prs
        }
        num_prs = len(records)

        # Check to see if we need to plot a log axis
        islog = False
        for rec, (pr, record) in enumerate(records.items()):
            islog |= check_log(record[self.name], ax[2])

        # Display the scatter plot for each PR
        for rec, (pr, record) in enumerate(records.items()):
            x = np.arange(len(record[self.name]))
            if islog:
                data = np.log10(record[self.name])
            else:
                data = record[self.name]
            ax[2].scatter(x, data, label=pr, s=10 * (num_prs - rec))
        ax[2].legend()
        ax[2].set_xlabel("blend index")

        # Load the records for all of the violin and box plot PRs
        records = {
            pr: np.load(os.path.join(__DATA_PATH__, set_id, get_filename(pr)))["records"]
            for pr in plot_prs
        }

        for ax_n, plot_type in enumerate(["box", "violin"]):
            # Extract the data
            x = np.arange(len(records))
            data = []
            for s, (pr, record) in enumerate(records.items()):
                data.append(record[self.name])

            # Check if we need a log plot
            islog = check_log(data, ax[ax_n])
            if islog:
                data = [np.log10(d) for d in data]

            if plot_type == "violin":
                # Make the violin plot
                ax[ax_n].violinplot(data, x, showmeans=False, showextrema=False, showmedians=False)

                # Calculate the quartile whiskers
                quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
                whiskers = np.array([
                    adjacent_values(sorted_array, q1, q3)
                    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
                whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
                # Display the whiskers
                ax[ax_n].scatter(x, medians, marker='o', color='white', s=30, zorder=3)
                ax[ax_n].vlines(x, quartile1, quartile3, color='k', linestyle='-', lw=5)
                ax[ax_n].vlines(x, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
            else:
                # Make the box plot
                ax[ax_n].boxplot(data)

        x_labels = tuple(records.keys())
        ax[1].xaxis.set_ticks(np.arange(len(x_labels)))
        ax[0].set_xticklabels(x_labels, size='small', rotation='vertical')
        ax[1].set_xticklabels(x_labels, size='small', rotation='vertical')

        ax[0].set_ylabel(self.units)
        fig.suptitle(self.name)


all_metrics = {
    "init time": Metric("init time", "time (ms)", False),
    "runtime": Metric("runtime", "time/source (ms)", False),
    "iterations": Metric("iterations", "iterations", False),
    "init logL": Metric("init logL", "logL", True),
    "logL": Metric("logL", "logL", True),
    "g diff": Metric("g diff", "truth-model", True),
    "r diff": Metric("r diff", "truth-model", True),
    "i diff": Metric("i diff", "truth-model", True),
    "z diff": Metric("z diff", "truth-model", True),
    "y diff": Metric("y diff", "truth-model", True),
}
