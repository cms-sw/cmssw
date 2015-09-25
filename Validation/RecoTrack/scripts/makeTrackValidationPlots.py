#!/usr/bin/env python

import os
import argparse

from Validation.RecoTrack.plotting.validation import SimpleValidation, SimpleSample
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots

def main(opts):
    files = opts.files
    labels = [f.replace(".root", "") for f in files]

    drawArgs={}
    if opts.ratio:
        drawArgs["ratio"] = True
    if opts.separate:
        drawArgs["separate"] = True
    if opts.png:
        drawArgs["saveFormat"] = ".png"

    val = SimpleValidation(files, labels, opts.outputDir)
    kwargs = {}
    if opts.html:
        htmlReport = val.createHtmlReport(validationName=opts.html_validation_name)
        htmlReport.beginSample(SimpleSample(opts.html_prefix, opts.html_sample))
        kwargs["htmlReport"] = htmlReport
    val.doPlots(trackingPlots.plotter, subdirprefix=opts.subdirprefix, plotterDrawArgs=drawArgs, **kwargs)
    val.doPlots(vertexPlots.plotter, subdirprefix=opts.subdirprefix, plotterDrawArgs=drawArgs, **kwargs)
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
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML pages")
    parser.add_argument("--html-prefix", default="plots",
                        help="Prefix for HTML page generation (default 'plots')")
    parser.add_argument("--html-sample", default="Sample",
                        help="Sample name for HTML page generation (default 'Sample')")
    parser.add_argument("--html-validation-name", default="",
                        help="Validation name for HTML page generation (enters to <title> element) (default '')")

    opts = parser.parse_args()
    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)

    if opts.ignoreMissing:
        print "--ignoreMissing is now the only operation mode, so you can stop using this parameter"

    main(opts)
