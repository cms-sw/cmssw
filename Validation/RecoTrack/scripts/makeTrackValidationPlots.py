#!/usr/bin/env python

import os
import argparse

import Validation.RecoTrack.plotting.plotting as plotting
from Validation.RecoTrack.plotting.validation import SimpleValidation
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots

def newdirname(algo, quality):
    return algo+"_"+quality

def subdirToAlgoQuality(subdir):
    return subdir.split("_")

def main(opts):
    files = opts.files
    labels = [f.replace(".root", "") for f in files]

    if opts.ignoreMissing:
        plotting.missingOk = True

    drawArgs={}
    if opts.ratio:
        drawArgs["ratio"] = True
    if opts.separate:
        drawArgs["separate"] = True
    if opts.png:
        drawArgs["saveFormat"] = ".png"

    val = SimpleValidation(files, labels, opts.outputDir)
    val.doPlotsAuto(trackingPlots.plotter, subdirToAlgoQuality=subdirToAlgoQuality, newdirFunc=newdirname, plotterDrawArgs=drawArgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create standard set of tracking validation plots from one or more DQM files. Note that the output directory structure is not exactly the same as with test/trackingPerformanceValidation.py or test/trackingCompare.py")
    parser.add_argument("files", metavar="file", type=str, nargs="+",
                        help="DQM file to plot the validation plots from")
    parser.add_argument("-o", "--outputDir", type=str, default="plots",
                        help="Plot output directory (default: 'plots')")
    parser.add_argument("--ignoreMissing", action="store_true",
                        help="Ignore missing histograms and directories")
    parser.add_argument("--ratio", action="store_true",
                        help="Create ratio pads")
    parser.add_argument("--separate", action="store_true",
                        help="Save all plots separately instead of grouping them")
    parser.add_argument("--png", action="store_true",
                        help="Save plots in PNG instead of PDF")

    opts = parser.parse_args()
    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)

    main(opts)
