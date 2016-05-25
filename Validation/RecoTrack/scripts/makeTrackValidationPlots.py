#!/usr/bin/env python

import os
import argparse

from Validation.RecoTrack.plotting.validation import SimpleValidation, SimpleSample
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots
import Validation.RecoTrack.plotting.plotting as plotting

class LimitTrackAlgo:
    def __init__(self, algos):
        self._algos = algos
    def __call__(self, algo, quality):
        return algo in self._algos

def limitRelVal(algo, quality):
    return quality in ["", "highPurity"]

def main(opts):
    sample = SimpleSample(opts.subdirprefix, opts.html_sample, [(f, f.replace(".root", "")) for f in opts.files])

    drawArgs={}
    if opts.ratio:
        drawArgs["ratio"] = True
    if opts.separate:
        drawArgs["separate"] = True
    if opts.png:
        drawArgs["saveFormat"] = ".png"
    if opts.verbose:
        plotting.verbose = True

    val = SimpleValidation([sample], opts.outputDir)
    kwargs = {}
    if opts.html:
        htmlReport = val.createHtmlReport(validationName=opts.html_validation_name)
        htmlReport.beginSample(sample)
        kwargs["htmlReport"] = htmlReport

    kwargs_tracking = {}
    kwargs_tracking.update(kwargs)
    if opts.limit_tracking_algo is not None:
        limitProcessing = LimitTrackAlgo(opts.limit_tracking_algo)
        kwargs_tracking["limitSubFoldersOnlyTo"] = {
            "": limitProcessing,
            "allTPEffic": limitProcessing,
            "fromPV": limitProcessing,
            "fromPVAllTP": limitProcessing,
            "seeding": limitProcessing,
            "building": limitProcessing,
        }
    if opts.limit_relval:
        ignore = lambda a,q: False
        kwargs_tracking["limitSubFoldersOnlyTo"] = {
            "": limitRelVal,
            "allTPEffic": ignore,
            "fromPV": ignore,
            "fromPVAllTP": ignore,
            "seeding": ignore,
            "building": ignore,
        }

    trk = [trackingPlots.plotter]
    other = [trackingPlots.timePlotter, vertexPlots.plotter]
    if opts.extended:
        trk.append(trackingPlots.plotterExt)
        other.append(vertexPlots.plotterExt)
    val.doPlots(trk, plotterDrawArgs=drawArgs, **kwargs_tracking)
    val.doPlots(other, plotterDrawArgs=drawArgs, **kwargs)
    print
    if opts.html:
        htmlReport.write()
        print "Plots and HTML report created into directory '%s'. You can just move it to some www area and access the pages via web browser" % opts.outputDir
    else:
        print "Plots created into directory '%s'." % opts.outputDir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create standard set of tracking validation plots from one or more DQM files.")
    parser.add_argument("files", metavar="file", type=str, nargs="+",
                        help="DQM file to plot the validation plots from")
    parser.add_argument("-o", "--outputDir", type=str, default="plots",
                        help="Plot output directory (default: 'plots')")
    parser.add_argument("--subdirprefix", type=str, default="plots",
                        help="Prefix for subdirectories inside outputDir (default: 'plots')")
    parser.add_argument("--ignoreMissing", action="store_true",
                        help="Ignore missing histograms and directories")
    parser.add_argument("--ratio", action="store_true",
                        help="Create ratio pads")
    parser.add_argument("--separate", action="store_true",
                        help="Save all plots separately instead of grouping them")
    parser.add_argument("--png", action="store_true",
                        help="Save plots in PNG instead of PDF")
    parser.add_argument("--limit-tracking-algo", type=str, default=None,
                        help="Comma separated list of tracking algos to limit to. (default: all algos; conflicts with --limit-relval)")
    parser.add_argument("--limit-relval", action="store_true",
                        help="Limit set of plots to those in release validation (almost). (default: all plots in the DQM files; conflicts with --limit-tracking-algo)")
    parser.add_argument("--extended", action="store_true",
                        help="Include extended set of plots (e.g. bunch of distributions; default off)")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML pages")
    parser.add_argument("--html-sample", default="Sample",
                        help="Sample name for HTML page generation (default 'Sample')")
    parser.add_argument("--html-validation-name", default="",
                        help="Validation name for HTML page generation (enters to <title> element) (default '')")
    parser.add_argument("--verbose", action="store_true",
                        help="Be verbose")

    opts = parser.parse_args()
    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)

    if opts.ignoreMissing:
        print "--ignoreMissing is now the only operation mode, so you can stop using this parameter"

    if opts.limit_tracking_algo is not None:
        if opts.limit_relval:
            parser.error("--limit-tracking-algo and --limit-relval conflict with each other")
        opts.limit_tracking_algo = opts.limit_tracking_algo.split(",")

    main(opts)
