#!/usr/bin/env python3

import os
import sys
import argparse
import ROOT

from dqm_plotting import split_csv, parse_limits, parse_rebin, load_hist, apply_rebin, plot_counts_and_rate, plot_comparison, make_sigma_over_mean_hist, invert_rate_hist


def load_root_object(file_path, hist_name, rebin=None, clone_suffix=""):
    root_file = ROOT.TFile.Open(file_path, "READ")

    if not root_file or root_file.IsZombie():
        print(f"ERROR: could not open {file_path}")
        return None

    obj = root_file.Get(hist_name)

    if not obj:
        print(f"WARNING: histogram not found: {hist_name}")
        root_file.Close()
        return None

    name = hist_name.split("/")[-1] + clone_suffix
    clone = obj.Clone(name + "_clone")
    clone.SetDirectory(0)

    if rebin is not None:
        clone = apply_rebin(clone, rebin, name)

    root_file.Close()
    return clone


def make_summary_plot(args):
    files = split_csv(args.files)
    den_hists = split_csv(args.den_hists)
    num_hists = split_csv(args.num_hists)
    rate_hists = split_csv(args.rate_hists)
    labels = split_csv(args.labels)

    if not (len(files) == len(den_hists) == len(num_hists) == len(rate_hists) == len(labels)):
        print("ERROR: --files, --den-hists, --num-hists, --rate-hists and --labels must have the same length.")
        print(f"files     : {len(files)}")
        print(f"den-hists : {len(den_hists)}")
        print(f"num-hists : {len(num_hists)}")
        print(f"rate-hists: {len(rate_hists)}")
        print(f"labels    : {len(labels)}")
        return False

    rebin = parse_rebin(args.rebin)

    made_any = False

    for i, (file_path, den_name, num_name, rate_name, label) in enumerate(zip(files, den_hists, num_hists, rate_hists, labels)):
        denominator = load_hist(file_path=file_path, hist_name=den_name, rebin=rebin, project_profile=True, clone_suffix=f"_den_{i}")
        numerator = load_hist(file_path=file_path, hist_name=num_name, rebin=rebin, project_profile=True, clone_suffix=f"_num_{i}")
        rate = load_hist(file_path=file_path, hist_name=rate_name, rebin=rebin, project_profile=True, clone_suffix=f"_rate_{i}")

        if denominator is None or numerator is None or rate is None:
            continue

        empty_bins = None
        if rate.InheritsFrom("TProfile"): empty_bins = [rate.GetBinEntries(ibin) == 0 for ibin in range(1, rate.GetNbinsX() + 1)]

        rate_inverted = rate.ProjectionX(rate.GetName() + "_proj") if rate.InheritsFrom("TProfile") else rate
        rate_inverted.SetDirectory(0)

        if args.inverted: invert_rate_hist(rate_inverted, empty_bins=empty_bins)

        if len(labels) == 1:
            output_name = args.name or rate_name.split("/")[-1]
        else:
            safe_label = label.replace("$", "").replace("\\", "").replace(" ", "_").replace("=", "").replace(".", "p")
            output_name = args.name or f"{rate_name.split('/')[-1]}_{safe_label}"

        output_path = os.path.join(args.odir, output_name + ".png")

        plot_counts_and_rate(
            denominator=denominator,
            numerator=numerator,
            rate=rate,
            output_path=output_path,
            denominator_label=args.den_label,
            numerator_label=args.num_label,
            rate_label=label,
            xlabel=args.xlabel,
            ylabel_rate=args.ylabel,
            cms_text=args.cms_text,
            energy_text=args.energy_text,
            xlim=parse_limits(args.xlim),
            right_ylim=parse_limits(args.ylim),
            right_log=args.logy,
            text=args.text,
            leg_title=args.leg_title,
        )

        made_any = True

    return made_any


def make_response_plot(args):
    files = split_csv(args.files)
    mean_hists = split_csv(args.mean_hists)
    sigma_hists = split_csv(args.sigma_hists)
    labels = split_csv(args.labels)

    if not (len(files) == len(mean_hists) == len(sigma_hists) == len(labels)):
        print("ERROR: --files, --mean-hists, --sigma-hists and --labels must have the same length.")
        print(f"files      : {len(files)}")
        print(f"mean-hists : {len(mean_hists)}")
        print(f"sigma-hists: {len(sigma_hists)}")
        print(f"labels     : {len(labels)}")
        return False

    rebin = parse_rebin(args.rebin)

    histograms = []
    good_labels = []

    for i, (file_path, mean_name, sigma_name, label) in enumerate(zip(files, mean_hists, sigma_hists, labels)):
        mean = load_root_object(file_path=file_path, hist_name=mean_name, rebin=rebin, clone_suffix=f"_mean_{i}")
        sigma = load_root_object(file_path=file_path, hist_name=sigma_name, rebin=rebin, clone_suffix=f"_sigma_{i}")

        if mean is None or sigma is None:
            continue

        h = make_sigma_over_mean_hist(sigma, mean, f"sigmaOverMean_{i}")
        h.SetTitle(args.title or "Response sigma/mean")

        histograms.append(h)
        good_labels.append(label)

    if not histograms:
        print("ERROR: no valid response histograms were loaded.")
        return False

    output_name = args.name or "Response_sigmaOverMean"
    output_path = os.path.join(args.odir, output_name + ".png")

    plot_comparison(
        histograms=histograms,
        labels=good_labels,
        output_path=output_path,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        xlim=parse_limits(args.xlim),
        ylim=parse_limits(args.ylim),
        ylim_ratio=parse_limits(args.ylim_ratio),
        logx=args.logx,
        logy=args.logy,
        cms_text=args.cms_text,
        energy_text=args.energy_text,
        leg_title=args.leg_title,
    )

    return True


def main():
    parser = argparse.ArgumentParser(description="Make Tau validation plots.")

    parser.add_argument("--mode", type=str, choices=["summary", "response"], required=True, help="summary: denominator/numerator/rate plot. response: sigma/mean comparison plot.")
    parser.add_argument("--files", type=str, required=True, help="Comma-separated list of input ROOT files.")
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated list of legend labels, one per input file.")

    # Summary mode
    parser.add_argument("--den-hists", type=str, default=None, help="Comma-separated list of denominator histogram paths.")
    parser.add_argument("--num-hists", type=str, default=None, help="Comma-separated list of numerator histogram paths.")
    parser.add_argument("--rate-hists", type=str, default=None, help="Comma-separated list of precomputed rate histogram paths.")
    parser.add_argument("--den-label", type=str, default="Denominator", help="Legend label for the denominator histogram.")
    parser.add_argument("--num-label", type=str, default="Numerator", help="Legend label for the numerator histogram.")
    parser.add_argument("--inverted", action="store_true", help="Invert histograms. E.g. Purity -> Fake Rates.")

    # Response mode
    parser.add_argument("--mean-hists", type=str, default=None, help="Comma-separated list of mean TProfile paths.")
    parser.add_argument("--sigma-hists", type=str, default=None, help="Comma-separated list of sigma TProfile paths.")

    # Common plotting options
    parser.add_argument("--odir", type=str, default="TauValidationPlots", help="Output directory where plots will be saved.")
    parser.add_argument("--name", type=str, default=None, help="Base name of the output plot. If omitted, a name is built automatically.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--rebin", type=str, default=None, help="Histogram rebinning. Use an integer factor, e.g. '2', or comma-separated bin edges, e.g. '0,20,40,60,100'.")
    parser.add_argument("--xlabel", type=str, required=True, help="X-axis label.")
    parser.add_argument("--ylabel", type=str, required=True, help="Y-axis label.")
    parser.add_argument("--xlim", type=str, default=None, help="X-axis range as 'xmin,xmax', e.g. '-2.5,2.5'.")
    parser.add_argument("--ylim", type=str, default=None, help="Y-axis range as 'ymin,ymax', e.g. '0,1.2'.")
    parser.add_argument("--ylim-ratio", type=str, default=None, help="Ratio-panel y-axis range as 'ymin,ymax', e.g. '0.5,1.5'.")
    parser.add_argument("--logx", action="store_true", help="Use a logarithmic scale for the x-axis.")
    parser.add_argument("--logy", action="store_true", help="Use a logarithmic scale for the y-axis.")
    parser.add_argument("--cms-text", type=str, default="Preliminary", help="CMS label text, e.g. 'Preliminary', 'Simulation' or an empty string.")
    parser.add_argument("--energy-text", type=str, default="", help="Additional CMS energy/luminosity text.")
    parser.add_argument("--text", type=str, default=None, help="Additional annotation text drawn on the plot.")
    parser.add_argument("--leg-title", type=str, default="", help="Title for the legend.")

    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)

    if args.mode == "summary":
        if args.den_hists is None or args.num_hists is None or args.rate_hists is None:
            print("ERROR: summary mode requires --den-hists, --num-hists and --rate-hists.")
            sys.exit(1)
        ok = make_summary_plot(args)

    elif args.mode == "response":
        if args.mean_hists is None or args.sigma_hists is None:
            print("ERROR: response mode requires --mean-hists and --sigma-hists.")
            sys.exit(1)
        ok = make_response_plot(args)

    else:
        print(f"ERROR: unknown mode {args.mode}")
        sys.exit(1)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
