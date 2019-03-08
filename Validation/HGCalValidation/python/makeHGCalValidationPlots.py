#!/usr/bin/env python

from __future__ import print_function
import os
import argparse

from Validation.RecoTrack.plotting.validation import SimpleValidation, SimpleSample
import Validation.HGCalValidation.hgcalPlots as hgcalPlots
import Validation.RecoTrack.plotting.plotting as plotting

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

    hgclayclus = [hgcalPlots.hgcalLayerClustersPlotter]
    val.doPlots(hgclayclus, plotterDrawArgs=drawArgs)
    print()
    if opts.no_html:
        print("Plots created into directory '%s'." % opts.outputDir)
    else:
        htmlReport.write()
        print("Plots and HTML report created into directory '%s'. You can just move it to some www area and access the pages via web browser" % opts.outputDir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create set of HGCal validation plots from one or more DQM files.")
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
    parser.add_argument("--no-html", action="store_true",
                        help="Disable HTML page genration")
    parser.add_argument("--html-sample", default="Sample",
                        help="Sample name for HTML page generation (default 'Sample')")
    parser.add_argument("--html-validation-name", default="",
                        help="Validation name for HTML page generation (enters to <title> element) (default '')")
    parser.add_argument("--verbose", action="store_true",
                        help="Be verbose")
    parser.add_argument("--lastLayerEEzm",type="int",default=28,
                        help="last layer of EE -z (default: %default)")
    parser.add_argument("--lastLayerFHzm",type="int",default=40,
                        help="last layer of FH -z (default: %default)")
    parser.add_argument("--maxlayerzm",type="int",default=52,
                        help="last layer of BH -z (default: %default)")
    parser.add_argument("--lastLayerEEzp",type="int",default=80,
                        help="last layer of EE +z (default: %default)")
    parser.add_argument("--lastLayerFHzp",type="int",default=92,
                        help="last layer of FH +z (default: %default)")
    parser.add_argument("--maxlayerzp",type="int",default=104,
                        help="last layer of BH +z (default: %default)")

    opts = parser.parse_args()

    #To be able to spot any issues both in -z and +z a layer id was introduced 
    #that spans from 0 to 103 for hgcal_v9 geometry. The mapping for hgcal_v9 is: 
    #-z: 0->51
    #+z: 52->103
    layerscheme = { 'lastLayerEEzm': opts.lastLayerEEzm, 'lastLayerFHzm': opts.lastLayerFHzm, 'maxlayerzm': opts.maxlayerzm, 
                    'lastLayerEEzp': opts.lastLayerEEzp, 'lastLayerFHzp': opts.lastLayerFHzp, 'maxlayerzp': opts.maxlayerzp }

    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)

    main(opts)
