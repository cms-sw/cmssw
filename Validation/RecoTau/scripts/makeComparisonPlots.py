#!/usr/bin/env python3

import os, sys, re
import ROOT
import argparse
import array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors as _mcolors
import mplhep as hep

# Set CMS style globally
plt.style.use(hep.style.CMS)

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
            "#3f8fda",
            "#ffa90e",
            "#bd1f01",
            "#94a4a2",
            "#832db6",
            "#a96b59",
            "#e76300",
            "#b9ac70",
            "#717581",
            "#92dadd",
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
            elif "turn-on" in title.lower():
                ylabel = "Turn-On"

        return (title, xlabel, ylabel)

    def _plot_histogram_data(self, ax, bin_centers, bin_contents, bin_errors, bin_edges, 
                           label, color_idx):
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

    def _calculate_and_plot_ratio(self, ax_ratio, bin_edges, bin_centers, bin_contents, bin_errors,
                                ref_centers, ref_contents, ref_errors, color_idx):
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

        ratio = np.divide(
            matching_contents,
            matching_ref_contents,
            out=np.zeros_like(matching_contents),
            where=matching_ref_contents != 0,
        )

        ratio_errors = np.zeros_like(ratio)
        mask = (matching_ref_contents != 0) & (matching_contents != 0)
        ratio_errors[mask] = np.abs(ratio[mask]) * np.sqrt(
            np.power(matching_errors[mask] / np.maximum(matching_contents[mask], 1e-10), 2)
            + np.power(matching_ref_errors[mask] / np.maximum(matching_ref_contents[mask], 1e-10), 2)
        )
        ratio_errors = np.nan_to_num(ratio_errors, nan=0.0, posinf=0.0, neginf=0.0)

        if len(matching_indices) > 0:
            curr_idxs, _ = zip(*matching_indices)
            curr_idxs = np.array(curr_idxs)

            # build correct edges from selected bins
            matching_edges = bin_edges[np.concatenate([
                curr_idxs,
                [curr_idxs[-1] + 1]
            ])]

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
                explicit_lines = lab.split("\\n")
                wrapped.append("\n".join(explicit_lines))
                continue
            
            # Preserve " - " separator (for overlay labels like "File - Collection")
            if " - " in lab:
                parts = lab.split(" - ", 1)  # Split only on first occurrence
                file_part = parts[0]
                collection_part = parts[1] if len(parts) > 1 else ""
                
                # If the combined length is too long, put on separate lines
                if len(lab) > width:
                    wrapped.append(f"{file_part}\n- {collection_part}")
                else:
                    wrapped.append(lab)
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
            wrapped_label = "\n".join(final_tokens) if final_tokens else lab
            wrapped.append(wrapped_label)
        return wrapped

    def _configure_legend(self, ax, labels, legend_title, place_outside=False):
        """Configure legend; wrap long entries and move outside if needed."""

        if not labels:
            return

        wrapped_labels = self._wrap_legend_labels(labels)

        legend_columns = len(wrapped_labels) if len(wrapped_labels) <= 3 else 3
        legend_fontsize = "20"
        if len(wrapped_labels) > 6:
            legend_fontsize = "18"
        if len(wrapped_labels) > 10:
            legend_fontsize = "16"

        if place_outside:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min, y_max + y_range * 0.1)
            ax.figure.subplots_adjust(right=0.9)
            ax.legend(
                wrapped_labels,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                title=legend_title,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
                frameon=False,
            )
            return

        ax.legend(
            wrapped_labels,
            loc="upper center",
            ncols=legend_columns,
            title=legend_title,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            columnspacing=1.0,
        )

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

    def plot_comparison(
        self,
        histograms,
        labels,
        output_path,
        x_lim=[None, None],
        y_lim=[None, None],
        y_lim_ratio=[None, None],
        xlabel=None,
        ylabel=None,
        logy=False,
        logx=False,
        cms_text="Preliminary",
        energy_text="",
    ):
        """
        Create comparison plot with ratio panel.

        Args:
            histograms: List of ROOT histograms to compare
            labels: List of labels for each histogram
            output_path: Output file path
            cms_text: CMS label text
            energy_text: Custom energy text (if None, uses default)
        """
        if len(histograms) != len(labels):
            raise ValueError("Number of histograms must match number of labels")

        if xlabel is None or ylabel is None:
            title, xlabel, ylabel = self.extract_labels_from_hist(histograms[0])

        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(
            2, 1, height_ratios=[1 - self.ratio_height, self.ratio_height], hspace=0.10
        )

        ax_main = fig.add_subplot(gs[0])

        # CMS styling
        hep.cms.label(cms_text, data=False, ax=ax_main, rlabel=energy_text, fontsize=20)

        ax_ratio = None
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

        ref_centers = None
        ref_contents = None
        ref_errors = None
        has_labels = False
        bin_labels = None

        all_ratios = []
        ref_centers = ref_contents = ref_errors = None
        
        for i, (hist, label) in enumerate(zip(histograms, labels)):
            bin_centers, bin_contents, bin_errors, bin_edges, bin_labels, has_labels = (
                self.root_to_numpy(hist)
            )

            # Use first histogram as reference
            if i == 0:
                ref_centers, ref_contents, ref_errors = bin_centers, bin_contents, bin_errors

            self._plot_histogram_data(ax_main, bin_centers, bin_contents, bin_errors, 
                                    bin_edges, label, i)

            if i > 0:
                valid_ratios = self._calculate_and_plot_ratio(
                    ax_ratio, bin_edges, bin_centers, bin_contents, bin_errors,
                    ref_centers, ref_contents, ref_errors, i)
                all_ratios.extend(valid_ratios)

        # Set custom labels if available
        if has_labels:
            ax_main.set_xticks(
                ref_centers,
                bin_labels,
                size="small" if len(bin_labels) < 10 else "xx-small",
                rotation=45,
                ha="right",
                va="top",
            )
            ax_main.tick_params(axis="x", which="minor", bottom=False)
        else:
            ax_main.set_xlabel(rf"{xlabel}", fontsize=20)

        ax_main.set_ylabel(rf"{ylabel}", fontsize=20)

        if x_lim[0] is not None and x_lim[1] is not None:
            ax_main.set_xlim(x_lim)
        
        if y_lim[0] is not None and y_lim[1] is not None:
            ax_main.set_ylim(y_lim)

        if logy:
            ax_main.set_yscale("log")
        if logx:
            ax_main.set_xscale("log")

        self._configure_legend(ax_main, labels, "")

        ax_main.grid(True, alpha=0.75, linestyle="dashdot", linewidth=0.75)

        self._apply_custom_formatter(ax_main)

        # Ratio plot styling
        if ax_ratio is not None:
            # Only show label in ratio and keep the same ticks as main plot
            ax_main.set_xlabel("")
            ax_main.tick_params(axis="x", labelbottom=False)

            # Set ratio plot limits
            y_min = y_lim_ratio[0] if y_lim_ratio else 0
            y_max = y_lim_ratio[1] if y_lim_ratio else 2
            ax_ratio.set_ylim(y_min, y_max)

            if has_labels:
                ax_ratio.set_xticks(
                    ref_centers,
                    bin_labels,
                    size="small" if len(bin_labels) < 10 else "xx-small",
                    rotation=45,
                    ha="right",
                    va="top",
                )
                ax_ratio.tick_params(axis="x", which="minor", bottom=False)
            else:
                ax_ratio.set_xlabel(xlabel, fontsize=20)

            ax_ratio.set_ylabel("Ratio", fontsize=20)
            ax_ratio.axhline(y=1, color="black", linestyle="--", alpha=0.7)

            ax_ratio.grid(True, alpha=0.75, linestyle="dashdot", linewidth=0.75)

        print(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

        pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make comparison plots.')
    parser.add_argument('--files', type=str, required=True,                                   help='Comma-separated list of DQM ROOT files.')
    parser.add_argument('--hists', type=str, required=True,                                   help='Comma-separated list of histogram names.')
    parser.add_argument('--labels', type=str, required=True,                                  help='Comma-separated list of legend labels.')
    parser.add_argument('--odir', type=str, default="ComparisonPlots", required=False,        help='Path to the output directory.')
    parser.add_argument('--name', type=str, default=None, required=False,                     help='Name of the output plot.')
    parser.add_argument('--rebin', type=str, default=None, required=False,                    help='Rebin for histograms (factor or vector).')
    parser.add_argument('--normalize', action='store_true',                                   help='Normalize histograms to unit area.')
    parser.add_argument('--xlim', type=str, default=None, required=False,                     help='X-axis limits (min,max).')
    parser.add_argument('--ylim', type=str, default=None, required=False,                     help='Y-axis limits (min,max).')
    parser.add_argument('--ylim-ratio', type=str, default=None, required=False,               help='Y-axis limits for ratio plot (min,max).')
    parser.add_argument('--logy', action='store_true',                                        help='Use logarithmic scale for y-axis.')
    parser.add_argument('--logx', action='store_true',                                        help='Use logarithmic scale for x-axis.')
    parser.add_argument('--xlabel', type=str, default=None, required=False,                   help='Custom x-axis label.')
    parser.add_argument('--ylabel', type=str, default=None, required=False,                   help='Custom y-axis label.')
    parser.add_argument('--cms-text', type=str, default="Preliminary", required=False,        help='CMS label text.')
    parser.add_argument('--energy-text', type=str, default=None, required=False,              help='Custom energy text for CMS label.')
    args = parser.parse_args()

    file_paths = [f.strip() for f in args.files.split(',')]
    hist_names = [h.strip() for h in args.hists.split(',')]
    label_names = [l.strip() for l in args.labels.split(',')]

    if len(file_paths) != len(hist_names) or len(file_paths) != len(label_names):
        print("Error: The number of files, histograms, and labels must be the same.")
        sys.exit(1)

    if args.odir:
        odir = args.odir
    else:
        odir = "./"

    plot_name = args.name if args.name else hist_names[0].split('/')[-1]
    
    plotter = DQMPlotter(figsize=(10, 10))

    histograms = []
    labels = []
    for i, (file_path, hist_name, label) in enumerate(zip(file_paths, hist_names, label_names)):
        root_file = ROOT.TFile.Open(file_path, "READ")
        if not root_file or root_file.IsZombie():
            print(f"Error: Could not open {file_path}")
            continue

        hist = root_file.Get(hist_name)
        hname = hist.GetName() if hist else "Unknown"

        # Clone histogram to avoid issues when file is closed
        hist_clone = hist.Clone(f"{hist}_clone_{i}")

        # Apply rebinning if requested
        if args.rebin is not None:
            if "," in args.rebin:
                rebin = [float(x.strip()) for x in args.rebin.split(',')]
            else:
                rebin = float(args.rebin)
            if isinstance(rebin, (int, float)):
                hist_clone = hist_clone.Rebin(int(rebin), hname + "_rebin_" + str(i))
            elif hasattr(rebin, '__iter__'):
                bin_edges_c = array.array('d', rebin)
                hist_clone = hist_clone.Rebin(len(bin_edges_c) - 1, hname + "_rebin_" + str(i), bin_edges_c)
            else:
                raise ValueError(f"Unknown type for rebin: {type(rebin)}")
        
        hist_clone.SetDirectory(0)
        
        if args.normalize:
            integral = hist_clone.Integral()
            if integral > 0:
                hist_clone.Scale(1.0 / integral)

        histograms.append(hist_clone)
        labels.append(label)

    if args.xlim is not None:
        x_lim_ = tuple(map(float, args.xlim.split(',')))
    else:
        x_lim_ = [None, None]
    if args.ylim is not None:
        y_lim_ = tuple(map(float, args.ylim.split(',')))
    else:
        y_lim_ = [None, None]
    if args.ylim_ratio is not None:
        y_lim_ratio_ = tuple(map(float, args.ylim_ratio.split(',')))
    else:
        y_lim_ratio_ = [None, None]

    plotter.plot_comparison(histograms, labels, odir+'/'+plot_name+'.png', 
                            x_lim=x_lim_, y_lim=y_lim_, y_lim_ratio=y_lim_ratio_,
                            xlabel=args.xlabel, ylabel=args.ylabel,
                            logy=args.logy, logx=args.logx, cms_text=args.cms_text, energy_text=args.energy_text, 
                            )
