#!/usr/bin/env python3

import os, re
import ROOT
import array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors as _mcolors
import mplhep as hep

# Set CMS style globally
plt.style.use(hep.style.CMS)


def split_csv(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_limits(text):
    if text is None:
        return [None, None]
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 2:
        raise ValueError(f"Expected min,max but got: {text}")
    return vals


def parse_rebin(text):
    if text is None:
        return None
    if "," in text:
        return [float(x.strip()) for x in text.split(",")]
    return int(float(text))


# From dqm-plot (class DQMPlotter is not importable)
class DQMPlotter:
    def __init__(self, figsize=(10, 10), ratio_height=0.3):
        """
        Initialize the plotter.

        Args:
            figsize: Figure size (width, height) in inches
            ratio_height: Fraction of figure height for ratio plot
        """
        self.figsize = figsize
        self.ratio_height = ratio_height
        # Color palette from https://cms-analysis.docs.cern.ch/guidelines/plotting/colors/
        self.colors = [
            "#832db6",
            "#3f8fda",
            "#a96b59",
            "#ffa90e",
            "#e76300",
            "#bd1f01",
            "#b9ac70",
            "#717581",
            "#92dadd",
            "#94a4a2",
        ]
        self.markers = [
            "o",  # circle
            "s",  # square
            "^",  # triangle up
            "D",  # diamond
            "v",  # triangle down
            "p",  # pentagon
            "*",  # star
            "h",  # hexagon
            "<",  # triangle left
            ">",  # triangle right
        ]

    def _clean_bin_label(self, label):
        """Clean ROOT bin label."""
        return label.strip() if label else ""

    def _extend_color_palette(self, needed: int):
        """Ensure self.colors has at least 'needed' distinct entries."""
        if needed <= len(self.colors):
            return

        extra_needed = needed - len(self.colors)
        new_colors = []

        cmap = plt.colormaps["hsv"]
        for i in range(max(extra_needed, 1)):
            rgba = cmap(i / max(extra_needed, 1))
            hexcol = _mcolors.to_hex(rgba, keep_alpha=False)
            if hexcol not in self.colors and hexcol not in new_colors:
                new_colors.append(hexcol)
            if len(new_colors) >= extra_needed:
                break

        self.colors.extend(new_colors)

    def apply_rebin(self, hist, rebin, name):
        if rebin is None:
            return hist

        if isinstance(rebin, int):
            h = hist.Rebin(rebin, name + "_rebin")
            h.SetDirectory(0)
            return h

        edges = array.array("d", rebin)
        h = hist.Rebin(len(edges) - 1, name + "_rebin", edges)
        h.SetDirectory(0)
        return h

    def project_if_profile(self, hist, name):
        if hist.InheritsFrom("TProfile"):
            h = hist.ProjectionX(name + "_proj")
            h.SetDirectory(0)
            return h
        return hist

    def load_hist(self, file_path, hist_name, rebin=None, project_profile=True, clone_suffix=""):
        root_file = ROOT.TFile.Open(file_path, "READ")

        if not root_file or root_file.IsZombie():
            print(f"ERROR: could not open {file_path}")
            return None

        hist = root_file.Get(hist_name)

        if not hist:
            print(f"WARNING: histogram not found: {hist_name}")
            root_file.Close()
            return None

        name = hist_name.split("/")[-1] + clone_suffix
        clone = hist.Clone(name + "_clone")
        clone.SetDirectory(0)

        clone = self.apply_rebin(clone, rebin, name)

        if project_profile:
            clone = self.project_if_profile(clone, name)

        root_file.Close()
        return clone

    def invert_rate_hist(self, hist, empty_bins=None):
        for ibin in range(1, hist.GetNbinsX() + 1):
            val = hist.GetBinContent(ibin)
            err = hist.GetBinError(ibin)

            # Empty original TProfile bin: no measurement
            if empty_bins is not None and empty_bins[ibin - 1]:
                hist.SetBinContent(ibin, 0.0)
                hist.SetBinError(ibin, 0.0)
                continue

            # Empty TH1-like bin: no measurement
            if empty_bins is None and val == 0.0 and err == 0.0:
                hist.SetBinContent(ibin, 0.0)
                hist.SetBinError(ibin, 0.0)
                continue

            hist.SetBinContent(ibin, 1.0 - val)
            hist.SetBinError(ibin, err)

    def normalise(self, hist):
        integral = hist.Integral()
        if integral > 0:
            hist.Scale(1.0 / integral)

    def root_to_numpy(self, hist):
        """
        Convert ROOT histogram to numpy arrays.

        Args:
            hist: ROOT histogram

        Returns:
            tuple: (bin_centers, bin_contents, bin_errors, bin_edges, bin_labels, has_labels)
        """
        n_bins = hist.GetNbinsX()
        bin_edges = np.array([hist.GetBinLowEdge(i) for i in range(1, n_bins + 2)])
        bin_centers = np.array([hist.GetBinCenter(i) for i in range(1, n_bins + 1)])
        bin_contents = np.array([hist.GetBinContent(i) for i in range(1, n_bins + 1)])
        bin_errors = np.array([hist.GetBinError(i) for i in range(1, n_bins + 1)])

        bin_labels = []
        for i in range(1, n_bins + 1):
            label = hist.GetXaxis().GetBinLabel(i)
            bin_labels.append(self._clean_bin_label(label) if label else "")

        # Only use extracted labels if meaningful
        has_labels = any(label and not label.isdigit() for label in bin_labels)

        return bin_centers, bin_contents, bin_errors, bin_edges, bin_labels, has_labels

    def extract_labels_from_hist(self, hist):
        """
        Extract title and axis labels from ROOT histogram.

        Args:
            hist: ROOT histogram

        Returns:
            tuple: (title, xlabel, ylabel)
        """
        title = hist.GetTitle()
        xlabel = title
        ylabel = "Occurrences"

        # Match "vs", "vs.", and flexible spacing/periods between v and s; allow underscores or spaces as delimiters
        vs_regex = re.compile(r'[_\s]+v\s*\.?\s*s\s*\.?[_\s]+', re.IGNORECASE)
        if vs_regex.search(title):
            # Detect explicit underscore-delimited form even with optional dots/spaces
            used_underscore_delim = bool(re.search(r'_v\s*\.?\s*s\s*\.?_', title.lower()))
            parts = vs_regex.split(title, maxsplit=1)
            left, right = parts[0].strip(), parts[1].strip()

            if used_underscore_delim:
                left = left.replace("_", " ")
                right = right.replace("_", " ")

            if "#sigma(" in title.lower():
                core = left[left.find("(") + 1 : left.rfind(")")]
                ylabel = r"$\delta$" + core + "/" + core
                right_clean = right
                if "Mean" in title:
                    ylabel = "<" + ylabel + ">"
                    right_clean = right_clean.replace("Mean", "")
                elif "Sigma" in title:
                    ylabel = r"$\sigma$(" + ylabel + ")"
                    right_clean = right_clean.replace("Sigma", "")
                xlabel = right_clean.strip()
            elif "Mean" in title:
                right_clean = right.replace("Mean", "").strip()
                ylabel = "<" + left + ">"
                xlabel = right_clean
            elif "Sigma" in title:
                right_clean = right.replace("Sigma", "").strip()
                ylabel = r"$\sigma$<" + left + ">"
                xlabel = right_clean
            else:
                # Default: "ylabel vs xlabel"
                ylabel = left
                xlabel = right
                if hist.InheritsFrom("TProfile") and "mean " in ylabel:
                    ylabel = ylabel.replace("mean ", "<") + ">"
        else:
            # Pull plots
            if "pull" not in title.lower():
                if "eta" in title.lower():
                    xlabel = r"$\eta$"
                elif "pt2" in title.lower():
                    xlabel = r"$p_{\mathrm{T}}^2$"
                elif "pt" in title.lower():
                    xlabel = r"$p_{\mathrm{T}}$"
                elif "phi" in title.lower():
                    xlabel = r"$\phi$"
            # Efficiency and turn-on plots
            if "eff" in title.lower():
                ylabel = "Efficiency"
            elif "fake" in title.lower():
                ylabel = "Fake rate"
            elif "dup" in title.lower():
                ylabel = "Duplicate rate"
            elif "split" in title.lower():
                ylabel = "Split rate"
            elif "turn-on" in title.lower():
                ylabel = "Turn-On"

        return title, xlabel, ylabel

    def _plot_histogram_data(self, ax, bin_centers, bin_contents, bin_errors, bin_edges, label, color_idx):
        """Plot histogram data either as histogram or error bars."""
        hep.histplot(
            bin_contents,
            bins=bin_edges,
            yerr=bin_errors,
            label=label,
            color=self.colors[color_idx % len(self.colors)],
            histtype="step",
            linewidth=2,
            ax=ax,
        )
        ax.errorbar(
            bin_centers,
            bin_contents,
            yerr=bin_errors,
            label=label,
            color=self.colors[color_idx % len(self.colors)],
            fmt=self.markers[color_idx % len(self.markers)],
            markersize=5,
            capsize=2,
            linewidth=1.5,
        )

    def _calculate_and_plot_ratio(self, ax_ratio, bin_edges, bin_centers, bin_contents, bin_errors, ref_centers, ref_contents, ref_errors, color_idx):
        """Calculate and plot ratio between current and reference histogram."""
        if ax_ratio is None:
            return []

        tolerance = 1e-6
        matching_indices = []

        for idx, center in enumerate(bin_centers):
            ref_idx = np.argmin(np.abs(ref_centers - center))
            if np.abs(ref_centers[ref_idx] - center) < tolerance:
                matching_indices.append((idx, ref_idx))

        if not matching_indices:
            return []

        curr_idxs, ref_idxs = zip(*matching_indices)
        matching_centers = bin_centers[list(curr_idxs)]
        matching_ref_contents = ref_contents[list(ref_idxs)]
        matching_contents = bin_contents[list(curr_idxs)]
        matching_ref_errors = ref_errors[list(ref_idxs)]
        matching_errors = bin_errors[list(curr_idxs)]

        ratio = np.divide(matching_contents, matching_ref_contents, out=np.zeros_like(matching_contents), where=matching_ref_contents != 0)

        ratio_errors = np.zeros_like(ratio)
        mask = (matching_ref_contents != 0) & (matching_contents != 0)
        ratio_errors[mask] = np.abs(ratio[mask]) * np.sqrt(
            np.power(matching_errors[mask] / np.maximum(matching_contents[mask], 1e-10), 2)
            + np.power(matching_ref_errors[mask] / np.maximum(matching_ref_contents[mask], 1e-10), 2)
        )
        ratio_errors = np.nan_to_num(ratio_errors, nan=0.0, posinf=0.0, neginf=0.0)

        curr_idxs = np.array(curr_idxs)

        # build correct edges from selected bins
        matching_edges = bin_edges[np.concatenate([curr_idxs, [curr_idxs[-1] + 1]])]

        ax_ratio.step(
            matching_edges,
            np.r_[ratio, ratio[-1]],  # extend last value for step plot
            where="post",
            color=self.colors[color_idx % len(self.colors)],
            linewidth=2,
            label="ratio",
        )

        ax_ratio.errorbar(
            matching_centers,
            ratio,
            yerr=ratio_errors,
            color=self.colors[color_idx % len(self.colors)],
            fmt=self.markers[color_idx % len(self.markers)],
            markersize=5,
            capsize=2,
            linewidth=1.5,
        )

        return ratio[(ratio > 0) & np.isfinite(ratio)]

    def _wrap_legend_labels(self, labels, width=25):
        """Soft-wrap legend labels at natural break points to reduce horizontal size."""
        wrapped = []
        for lab in labels:
            # Explicit new lines
            if "\\n" in lab:
                wrapped.append("\n".join(lab.split("\\n")))
                continue

            # Preserve " - " separator (for overlay labels like "File - Collection")
            if " - " in lab:
                parts = lab.split(" - ", 1)  # Split only on first occurrence
                file_part = parts[0]
                collection_part = parts[1] if len(parts) > 1 else ""

                # If the combined length is too long, put on separate lines
                wrapped.append(f"{file_part}\n- {collection_part}" if len(lab) > width else lab)
                continue

            # Automatic split using / or _
            parts = re.split(r'(/|_)+', lab)
            tokens = []
            buffer = ""
            for p in parts:
                if not p:
                    continue
                candidate = (buffer + p) if buffer else p
                if len(candidate) > width and buffer:
                    tokens.append(buffer.rstrip("_/"))
                    buffer = p
                else:
                    buffer = candidate
            if buffer:
                tokens.append(buffer.rstrip("_/"))

            # CamelCase and digit splitting
            final_tokens = []
            for tok in tokens:
                if len(tok) > width:
                    subtoks = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+', tok)
                    line = ""
                    for st in subtoks:
                        if len(line) + len(st) + 1 > width and line:
                            final_tokens.append(line)
                            line = st
                        else:
                            line = (line + st) if not line else (line + st)
                    if line:
                        final_tokens.append(line)
                else:
                    final_tokens.append(tok)

            wrapped.append("\n".join(final_tokens) if final_tokens else lab)

        return wrapped

    def _configure_legend(self, ax, labels, legend_title, place_outside=False):
        """Configure legend; wrap long entries and move outside if needed."""
        if not labels:
            return

        wrapped_labels = self._wrap_legend_labels(labels)

        legend_columns = len(wrapped_labels) if len(wrapped_labels) <= 3 else 3
        legend_fontsize = "20"
        if len(wrapped_labels) > 3:
            legend_fontsize = "18"
        if len(wrapped_labels) > 10:
            legend_fontsize = "16"

        if place_outside:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min, y_max + y_range * 0.1)
            ax.figure.subplots_adjust(right=0.9)
            ax.legend(wrapped_labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, title=legend_title, fontsize=legend_fontsize, title_fontsize=legend_fontsize, frameon=False)
            return

        ax.legend(wrapped_labels, loc="upper center", ncols=legend_columns, title=legend_title, fontsize=legend_fontsize, title_fontsize=legend_fontsize, columnspacing=1.0, frameon=False)

    def _apply_custom_formatter(self, ax):
        """Apply custom scientific notation formatter to y-axis."""
        if ax.get_yscale() != "log":
            from matplotlib.ticker import ScalarFormatter

            class CustomScalarFormatter(ScalarFormatter):
                def format_data_short(self, value):
                    if self.orderOfMagnitude != 0:
                        return f"×10$^{{{self.orderOfMagnitude}}}$"
                    return ""

            formatter = CustomScalarFormatter(useOffset=True, useMathText=True)
            ax.yaxis.set_major_formatter(formatter)

            # Move y-axis scientific notation to avoid overlap with CMS label
            ax.yaxis.get_offset_text().set_position((-0.01, 1.02))
            ax.yaxis.get_offset_text().set_horizontalalignment("right")
            ax.yaxis.get_offset_text().set_verticalalignment("bottom")

    def plot_comparison(self, histograms, labels, output_path, 
        x_lim=[None, None], y_lim=[None, None], y_lim_ratio=[None, None],
        xlabel=None, ylabel=None, leg_title="", logy=False, logx=False, cms_text="Preliminary", energy_text=""):
        """
        Create comparison plot with ratio panel.

        Args:
            histograms: List of ROOT histograms to compare
            labels: List of labels for each histogram
            output_path: Output file path
            cms_text: CMS label text
            energy_text: Custom energy text (if None, uses default)
        """
        if len(histograms) == 0:
            raise RuntimeError("No histograms to plot.")
        if len(histograms) != len(labels):
            raise ValueError("Number of histograms must match number of labels")

        self._extend_color_palette(len(histograms))

        if xlabel is None or ylabel is None:
            _, auto_x, auto_y = self.extract_labels_from_hist(histograms[0])
            xlabel = xlabel or auto_x
            ylabel = ylabel or auto_y

        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1 - self.ratio_height, self.ratio_height], hspace=0.10)

        ax_main = fig.add_subplot(gs[0])

        # CMS styling
        hep.cms.label(cms_text, data=False, ax=ax_main, rlabel=energy_text or "", fontsize=20)

        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

        ref_centers = None
        ref_contents = None
        ref_errors = None
        ref_labels = None
        has_bin_labels = False

        all_ratios = []
        main_upper = []
        all_ratio_upper = []
        all_ratio_lower = []

        for i, (hist, label) in enumerate(zip(histograms, labels)):
            bin_centers, bin_contents, bin_errors, bin_edges, bin_labels, has_labels = self.root_to_numpy(hist)

            main_upper.extend(bin_contents + bin_errors)

            if i == 0:
                ref_centers = bin_centers
                ref_contents = bin_contents
                ref_errors = bin_errors
                ref_labels = bin_labels
                has_bin_labels = has_labels

            self._plot_histogram_data(ax_main, bin_centers, bin_contents, bin_errors, bin_edges, label, i)

            if i > 0:
                valid_ratios = self._calculate_and_plot_ratio(ax_ratio, bin_edges, bin_centers, bin_contents, bin_errors, ref_centers, ref_contents, ref_errors, i)
                all_ratios.extend(valid_ratios)

                valid = ref_contents != 0
                ratio_values = np.divide(bin_contents, ref_contents, out=np.zeros_like(bin_contents, dtype=float), where=valid)
                ratio_errors = np.zeros_like(ratio_values, dtype=float)
                ratio_errors[valid] = np.sqrt((bin_errors[valid] / ref_contents[valid]) ** 2 + ((bin_contents[valid] / ref_contents[valid]) * (ref_errors[valid] / ref_contents[valid])) ** 2)
                all_ratio_upper.extend(ratio_values[valid] + ratio_errors[valid])
                all_ratio_lower.extend(ratio_values[valid] - ratio_errors[valid])

        # Set custom labels if available
        if has_bin_labels:
            ax_main.set_xticks(ref_centers)
            ax_main.set_xticklabels(ref_labels, size="small" if len(ref_labels) < 10 else "xx-small", rotation=45, ha="right", va="top")
            ax_main.tick_params(axis="x", which="minor", bottom=False)
        else:
            ax_main.set_xlabel(rf"{xlabel}", fontsize=20)

        ax_main.set_ylabel(rf"{ylabel}", fontsize=20)

        if x_lim[0] is not None and x_lim[1] is not None:
            ax_main.set_xlim(x_lim)

        if y_lim[0] is not None and y_lim[1] is not None:
            ax_main.set_ylim(y_lim)
        else:
            ymax = 1.6 * np.max(main_upper) if len(main_upper) and np.max(main_upper) > 0 else 1.0
            ymin = max(1e-3, 0.5 * np.min([x for x in main_upper if x > 0])) if logy and any(x > 0 for x in main_upper) else 0.0
            ax_main.set_ylim(ymin, ymax)

        if logy:
            ax_main.set_yscale("log")
        if logx:
            ax_main.set_xscale("log")

        self._configure_legend(ax_main, labels, leg_title)

        ax_main.grid(True, alpha=0.75, linestyle="dashdot", linewidth=0.75)

        self._apply_custom_formatter(ax_main)

        # Ratio plot styling
        ax_main.set_xlabel("")
        ax_main.tick_params(axis="x", labelbottom=False)

        # Set ratio plot limits
        if y_lim_ratio[0] is not None and y_lim_ratio[1] is not None:
            ax_ratio.set_ylim(y_lim_ratio)
        else:
            ratio_min = np.min(all_ratio_lower) if len(all_ratio_lower) else 0.0
            ratio_max = np.max(all_ratio_upper) if len(all_ratio_upper) else 2.0
            span = ratio_max - ratio_min
            scale = max(abs(ratio_min), abs(ratio_max), 1.0)
            pad = max(0.20 * span, 0.05 * scale**0.5)
            if span == 0:
                pad = 0.10 * scale**0.5
            ax_ratio.set_ylim(ratio_min - pad, ratio_max + pad)

        if has_bin_labels:
            ax_ratio.set_xticks(ref_centers)
            ax_ratio.set_xticklabels(ref_labels, size="small" if len(ref_labels) < 10 else "xx-small", rotation=45, ha="right", va="top")
            ax_ratio.tick_params(axis="x", which="minor", bottom=False)
        else:
            ax_ratio.set_xlabel(xlabel, fontsize=20)

        ax_ratio.set_ylabel("Ratio", fontsize=20)
        ax_ratio.axhline(y=1, color="black", linestyle="--", alpha=0.7)
        ax_ratio.grid(True, alpha=0.75, linestyle="dashdot", linewidth=0.75)

        outdir = os.path.dirname(output_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        print(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

        pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
        print(pdf_path)
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close()

    def make_sigma_over_mean_hist(self, sigma_profile, mean_profile, name):
        sigma = self.project_if_profile(sigma_profile, name + "_sigma")
        mean = self.project_if_profile(mean_profile, name + "_mean")
        out = sigma.Clone(name)
        out.Reset("ICES")
        out.SetDirectory(0)
        for ibin in range(1, sigma.GetNbinsX() + 1):
            s = sigma.GetBinContent(ibin)
            m = mean.GetBinContent(ibin)
            s_err = sigma.GetBinError(ibin)
            m_err = mean.GetBinError(ibin)
            if m == 0: # undefined ratio
                out.SetBinContent(ibin, 0.0)
                out.SetBinError(ibin, 0.0)
                continue
            value = s / m
            err = np.sqrt((s_err / m) ** 2 + ((s / m) * (m_err / m)) ** 2)
            out.SetBinContent(ibin, value)
            out.SetBinError(ibin, err)
        return out

    def plot_counts_and_rate(self, denominator, numerator, rate, output_path, denominator_label, numerator_label, rate_label, xlabel, ylabel_rate, cms_text="Preliminary", energy_text="", xlim=(None, None), right_ylim=(0.0, 1.25), right_log=False, text=None, leg_title=""):
        centres, rate_values, rate_errors, edges, _, _ = self.root_to_numpy(rate)
        _, den_values, den_errors, _, _, _ = self.root_to_numpy(denominator)
        _, num_values, num_errors, _, _, _ = self.root_to_numpy(numerator)

        widths = np.diff(edges)

        fig, ax = plt.subplots(figsize=self.figsize)

        hep.cms.label(cms_text, data=False, ax=ax, rlabel=energy_text or "", fontsize=20)

        den_step = np.r_[den_values, den_values[-1]]
        num_step = np.r_[num_values, num_values[-1]]
        ax.step(edges, den_step, where="post", label=denominator_label, color="black", linewidth=2)
        ax.step(edges, num_step, where="post", label=numerator_label, color="#9c9ca1", linestyle="-.", linewidth=2)
        ax.fill_between(edges, num_step, step="post", alpha=0.3, color="#9c9ca1")

        # print("last den:", den_values[-1], "last num:", num_values[-1], "last rate:", rate_values[-1], "last edges:", edges[-2], edges[-1])

        # Set an automatic ymax based on the max value + the up error
        counts_upper = np.concatenate([den_values + den_errors, num_values + num_errors])
        ymax = 1.2 * np.max(counts_upper) if len(counts_upper) and np.max(counts_upper) > 0 else 1.0
        rate_upper = rate_values + rate_errors
        rate_ymax = 1.25 * np.max(rate_upper) if len(rate_upper) and np.max(rate_upper) > 0 else right_ylim[1]

        if xlim[0] is not None and xlim[1] is not None:
            ax.set_xlim(xlim)
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel("Entries", fontsize=20)
        leg = ax.legend(loc="upper left", frameon=False, fontsize=18, title=leg_title, title_fontsize=18,
                           borderaxespad=0.5, handletextpad=0.8)
        leg._legend_box.align = "left"
        leg.get_title().set_ha("left")

        ax2 = ax.twinx()
        ax2.set_ylabel(ylabel_rate, color="#bd1f01", fontsize=20)
        if right_ylim[0] is not None and right_ylim[1] is not None:
            ax2.set_ylim(right_ylim)
        else:
            ax2.set_ylim(0.0, rate_ymax)

        if right_log:
            ax2.set_yscale("log")

        ax2.errorbar(centres, rate_values, xerr=0.5 * widths, yerr=rate_errors, fmt="o", color="#bd1f01", capsize=2, linewidth=1.5, label=rate_label)
        # ax2.axhline(y=1.0, color="#bd1f01", linewidth=2, linestyle="--", alpha=0.7)

        ax.grid(True, axis="x", alpha=0.7, linestyle="dashdot")
        ax2.grid(True, axis="y", alpha=0.7, linestyle="dashdot")
        ax2.tick_params(axis="y", labelcolor="#bd1f01")

        if text:
            ax.text(0.97, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=18)

        outdir = os.path.dirname(output_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        print(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

        pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
        print(pdf_path)
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close()


_default_plotter = DQMPlotter()


def apply_rebin(hist, rebin, name):
    return _default_plotter.apply_rebin(hist, rebin, name)


def project_if_profile(hist, name):
    return _default_plotter.project_if_profile(hist, name)


def load_hist(file_path, hist_name, rebin=None, project_profile=True, clone_suffix=""):
    return _default_plotter.load_hist(file_path, hist_name, rebin, project_profile, clone_suffix)


def invert_rate_hist(hist, empty_bins=None):
    return _default_plotter.invert_rate_hist(hist, empty_bins)


def normalise(hist):
    return _default_plotter.normalise(hist)


def hist_to_numpy(hist):
    return _default_plotter.root_to_numpy(hist)


def make_sigma_over_mean_hist(sigma_profile, mean_profile, name):
    return _default_plotter.make_sigma_over_mean_hist(sigma_profile, mean_profile, name)


def plot_comparison(histograms, labels, output_path, xlabel=None, ylabel=None, xlim=None, ylim=None, ylim_ratio=None, leg_title="", logx=False, logy=False, cms_text="Preliminary", energy_text=""):
    return _default_plotter.plot_comparison(histograms, labels, output_path, x_lim=xlim or [None, None], y_lim=ylim or [None, None], y_lim_ratio=ylim_ratio or [None, None], xlabel=xlabel, ylabel=ylabel, leg_title=leg_title, logx=logx, logy=logy, cms_text=cms_text, energy_text=energy_text)


def plot_counts_and_rate(denominator, numerator, rate, output_path, denominator_label, numerator_label, rate_label, xlabel, ylabel_rate, cms_text="Preliminary", energy_text="", xlim=(None, None), right_ylim=(0.0, 1.25), right_log=False, text=None, leg_title=""):
    return _default_plotter.plot_counts_and_rate(denominator, numerator, rate, output_path, denominator_label, numerator_label, rate_label, xlabel, ylabel_rate, cms_text, energy_text, xlim, right_ylim, right_log, text, leg_title)