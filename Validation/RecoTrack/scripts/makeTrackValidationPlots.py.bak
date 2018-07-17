#!/usr/bin/env python

import os
import argparse

from Validation.RecoTrack.plotting.validation import SimpleValidation, SimpleSample
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots
import Validation.RecoTrack.plotting.plotting as plotting

class LimitTrackAlgo:
    def __init__(self, algos, includePtCut):
        self._algos = algos
        self._includePtCut = includePtCut
    def __call__(self, algo, quality):
        if self._algos is not None and algo not in self._algos:
            return False
        if not self._includePtCut and "Pt09" in quality:
            return False
        return True

def limitRelVal(algo, quality):
    return quality in ["", "highPurity", "ByOriginalAlgo", "highPurityByOriginalAlgo"]

def main(opts):
    sample = SimpleSample(opts.subdirprefix, opts.html_sample, [(f, f.replace(".root", "")) for f in opts.files])

    drawArgs={}
    if opts.no_ratio:
        drawArgs["ratio"] = False
    if opts.separate:
        drawArgs["separate"] = True
    if opts.png:
        drawArgs["saveFormat"] = ".png"
    if opts.verbose:
        plotting.verbose = True

    val = SimpleValidation([sample], opts.outputDir)
    htmlReport = val.createHtmlReport(validationName=opts.html_validation_name)

    limitProcessing = LimitTrackAlgo(opts.limit_tracking_algo, includePtCut=opts.ptcut)
    kwargs_tracking = {
        "limitSubFoldersOnlyTo": {
            "": limitProcessing,
            "allTPEffic": limitProcessing,
            "fromPV": limitProcessing,
            "fromPVAllTP": limitProcessing,
            "tpPtLess09": limitProcessing,
            "seeding": limitProcessing,
            "building": limitProcessing,
            "bhadron": limitProcessing,
        }
    }
    if opts.limit_relval:
        ignore = lambda a,q: False
        kwargs_tracking["limitSubFoldersOnlyTo"] = {
            "": limitRelVal,
            "allTPEffic": ignore,
            "fromPV": ignore,
            "fromPVAllTP": ignore,
            "tpPtLess09": limitRelVal,
            "seeding": ignore,
            "bhadron": limitRelVal,
        }

    trk = [trackingPlots.plotter]
    other = [trackingPlots.timePlotter, vertexPlots.plotter, trackingPlots.plotterHLT]
    if opts.extended:
        trk.append(trackingPlots.plotterExt)
        other.extend([vertexPlots.plotterExt, trackingPlots.plotterHLTExt])
    val.doPlots(trk, plotterDrawArgs=drawArgs, **kwargs_tracking)
    val.doPlots(other, plotterDrawArgs=drawArgs)
    print
    if opts.no_html:
        print "Plots created into directory '%s'." % opts.outputDir
    else:
        htmlReport.write()
        print "Plots and HTML report created into directory '%s'. You can just move it to some www area and access the pages via web browser" % opts.outputDir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create standard set of tracking validation plots from one or more DQM files.")
    parser.add_argument("files", metavar="file", type=str, nargs="+",
                        help="DQM file to plot the validation plots from")
    parser.add_argument("-o", "--outputDir", type=str, default="plots",
                        help="Plot output directory (default: 'plots')")
    parser.add_argument("--subdirprefix", type=str, default="plots",
                        help="Prefix for subdirectories inside outputDir (default: 'plots')")
    parser.add_argument("--no-ratio", action="store_true",
                        help="Disable ratio pads")
    parser.add_argument("--separate", action="store_true",
                        help="Save all plots separately instead of grouping them")
    parser.add_argument("--png", action="store_true",
                        help="Save plots in PNG instead of PDF")
    parser.add_argument("--limit-tracking-algo", type=str, default=None,
                        help="Comma separated list of tracking algos to limit to. (default: all algos; conflicts with --limit-relval)")
    parser.add_argument("--limit-relval", action="store_true",
                        help="Limit set of plots to those in release validation (almost). (default: all plots in the DQM files; conflicts with --limit-tracking-algo)")
    parser.add_argument("--ptcut", action="store_true",
                        help="Include plots with pT > 0.9 GeV cut (with --limit-relval, does not have any effect)")
    parser.add_argument("--extended", action="store_true",
                        help="Include extended set of plots (e.g. bunch of distributions; default off)")
    parser.add_argument("--no-html", action="store_true",
                        help="Disable HTML page genration")
    parser.add_argument("--html-sample", default="Sample",
                        help="Sample name for HTML page generation (default 'Sample')")
    parser.add_argument("--html-validation-name", default="",
                        help="Validation name for HTML page generation (enters to <title> element) (default '')")
    parser.add_argument("--verbose", action="store_true",
                        help="Be verbose")

    group = parser.add_argument_group("deprecated arguments (they have no effect and will be removed in the future):")
    group.add_argument("--ignoreMissing", action="store_true",
                       help="Ignore missing histograms and directories (deprecated, is this is already the default mode)")
    group.add_argument("--ratio", action="store_true",
                       help="Create ratio pads (deprecated, as it is already the default")
    group.add_argument("--html", action="store_true",
                       help="Generate HTML pages (deprecated, as it is already the default")

    opts = parser.parse_args()
    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)

    if opts.ignoreMissing:
        print "--ignoreMissing is now the only operation mode, so you can stop using this parameter"

    if opts.ratio:
        print "--ratio is now the default, so you can stop using this parameter"

    if opts.html:
        print "--html is now the default, so you can stop using this parameter"

    if opts.limit_tracking_algo is not None:
        if opts.limit_relval:
            parser.error("--limit-tracking-algo and --limit-relval conflict with each other")
        opts.limit_tracking_algo = opts.limit_tracking_algo.split(",")

    if opts.limit_relval and opts.ptcut:
        print "With --limit-relval enabled, --ptcut option does not have any effect"

    main(opts)
