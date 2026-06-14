#!/usr/bin/env python3

import os
import sys
import argparse

from dqm_plotting import split_csv, parse_limits, parse_rebin, load_hist, invert_rate_hist, normalise, plot_comparison


def main():
    parser = argparse.ArgumentParser(description="Make comparison plots.")

    parser.add_argument("--files", type=str, required=True, help="Comma-separated list of DQM ROOT files.")
    parser.add_argument("--hists", type=str, required=True, help="Comma-separated list of histogram names.")
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated list of legend labels.")
    parser.add_argument("--odir", type=str, default="ComparisonPlots", help="Path to the output directory.")
    parser.add_argument("--name", type=str, default=None, help="Name of the output plot.")
    parser.add_argument("--rebin", type=str, default=None, help="Rebin for histograms, factor or vector.")
    parser.add_argument("--normalize", action="store_true", help="Normalize histograms to unit area.")
    parser.add_argument("--xlim", type=str, default=None, help="X-axis limits, min,max.")
    parser.add_argument("--ylim", type=str, default=None, help="Y-axis limits, min,max.")
    parser.add_argument("--ylim-ratio", type=str, default=None, help="Y-axis limits for ratio plot, min,max.")
    parser.add_argument("--leg-title", type=str, default="", help="Title for the legend.")
    parser.add_argument("--logy", action="store_true", help="Use logarithmic scale for y-axis.")
    parser.add_argument("--logx", action="store_true", help="Use logarithmic scale for x-axis.")
    parser.add_argument("--xlabel", type=str, default=None, help="Custom x-axis label.")
    parser.add_argument("--ylabel", type=str, default=None, help="Custom y-axis label.")
    parser.add_argument("--cms-text", type=str, default="Preliminary", help="CMS label text.")
    parser.add_argument("--energy-text", type=str, default=None, help="Custom energy text for CMS label.")
    parser.add_argument("--inverted", action="store_true", help="Invert histograms. E.g. Purity -> Fake Rates.")

    args = parser.parse_args()

    files = split_csv(args.files)
    hists = split_csv(args.hists)
    labels = split_csv(args.labels)

    if not (len(files) == len(hists) == len(labels)):
        print("ERROR: --files, --hists and --labels must have the same length.")
        print(f"files : {len(files)}")
        print(f"hists : {len(hists)}")
        print(f"labels: {len(labels)}")
        sys.exit(1)

    rebin = parse_rebin(args.rebin)

    histograms = []
    good_labels = []

    for i, (file_path, hist_name, label) in enumerate(zip(files, hists, labels)):
        h = load_hist(file_path=file_path, hist_name=hist_name, rebin=rebin, project_profile=False, clone_suffix=f"_{i}")

        if h is None:
            continue

        empty_bins = None
        if h.InheritsFrom("TProfile"):
            empty_bins = [ h.GetBinEntries(ibin) == 0 for ibin in range(1, h.GetNbinsX() + 1) ]

        h = h.ProjectionX(h.GetName() + "_proj") if h.InheritsFrom("TProfile") else h
        h.SetDirectory(0)

        if args.inverted:
            invert_rate_hist(h, empty_bins=empty_bins)

        if args.normalize:
            normalise(h)

        histograms.append(h)
        good_labels.append(label)

    if not histograms:
        print("ERROR: no valid histograms were loaded.")
        sys.exit(1)

    plot_name = args.name or hists[0].split("/")[-1]
    output_path = os.path.join(args.odir, plot_name + ".png")

    plot_comparison(
        histograms=histograms,
        labels=good_labels,
        output_path=output_path,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        xlim=parse_limits(args.xlim),
        ylim=parse_limits(args.ylim),
        ylim_ratio=parse_limits(args.ylim_ratio),
        leg_title=args.leg_title,
        logx=args.logx,
        logy=args.logy,
        cms_text=args.cms_text,
        energy_text=args.energy_text,
    )


if __name__ == "__main__":
    main()
