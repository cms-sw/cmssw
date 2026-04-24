#!/usr/bin/env python3

import mplhep as mpl
mpl.style.use(mpl.style.CMS)
import uproot, hist
import matplotlib.pyplot as plt
import numpy as np
import argparse

def make_eff_fake(h_eff, h_fake):
    eff, fakes = [], []
    for pidCut in h_eff.axes[0].edges:
        eff.append(h_eff[hist.loc(pidCut)::hist.sum] / h_eff[hist.sum] if h_eff[hist.sum]>0 else 0.)
        fakes.append(h_fake[hist.loc(pidCut)::hist.sum] / h_fake[hist.sum] if h_fake[hist.sum] else 0.)
    return h_eff.axes[0].edges, eff, fakes

prefix = "DQMData/Run 1/HGCAL/Run summary/TICLTracksterPIDValidation/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='makeTiclSuperclusteringPIDValidationPlots',
                    description='Takes multidimensional histograms from DQM output and makes ROC curves for PID used for superclustering')
    parser.add_argument('filename', nargs="+", help="DQM files to do the plots") 
    args = parser.parse_args()

    # ROC curve for seed
    plt.figure(figsize=(7, 7))
    for file in args.filename:
        with uproot.open(file) as f:
            pid, eff, fake = make_eff_fake(f[prefix+"superclusteringSeedTrackster/pt_eta_pid"].to_hist()[hist.sum, hist.loc(1.6):hist.loc(2.9):hist.sum, :], f[prefix+"superclusteringSeedTrackster/pt_eta_pid_fakes"].to_hist()[hist.sum, hist.loc(1.6):hist.loc(2.9):hist.sum, :])
        plt.plot(eff, fake, "o-",  markersize=4, label=file)

    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.xlim(0.8, 1.)
    plt.ylim(1e-8, 2)
    plt.yscale("log")
    plt.legend(fontsize=15)
    plt.text(0.04, 0.96, "Seed electron trackster", transform=plt.gca().transAxes, va="top", ha="left", fontsize=20)
    plt.savefig("superclusteringSeedTrackster_ROC.png", bbox_inches="tight")
    plt.savefig("superclusteringSeedTrackster_ROC.pdf", bbox_inches="tight")

    # ROC curve for candidate
    plt.figure(figsize=(7, 7))
    for file in args.filename:
        with uproot.open(file) as f:
            pid, eff, fake = make_eff_fake(f[prefix+"superclusteringCandidateTrackster/pt_eta_pid"].to_hist()[hist.sum, hist.loc(1.6):hist.loc(2.9):hist.sum, :], f[prefix+"superclusteringCandidateTrackster/pt_eta_pid_fakes"].to_hist()[hist.sum,  hist.loc(1.6):hist.loc(2.9):hist.sum, :])
        plt.plot(eff, fake, "o-", markersize=4, label=file)

    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.legend(fontsize=15)
    plt.text(0.04, 0.96, "Trackster from electron 'brem'", transform=plt.gca().transAxes, va="top", ha="left", fontsize=20)
    plt.savefig("superclusteringCandidateTrackster_ROC.png", bbox_inches="tight")
    plt.savefig("superclusteringCandidateTrackster_ROC.pdf", bbox_inches="tight")

