#!/usr/bin/env python

from __future__ import print_function
import os
import argparse

from Validation.RecoTrack.plotting.validation import SimpleValidation, SimpleSample
import Validation.HGCalValidation.hgcalPlots as hgcalPlots
import Validation.RecoTrack.plotting.plotting as plotting

def main(opts):

    drawArgs={}
    if opts.no_ratio:
        drawArgs["ratio"] = False
    if opts.separate:
        drawArgs["separate"] = True
    if opts.png:
        drawArgs["saveFormat"] = ".png"
    if opts.verbose:
        plotting.verbose = True

    filenames = [(f, f.replace(".root", "")) for f in opts.files]
    sample = SimpleSample(opts.subdirprefix[0], opts.html_sample, filenames)
  
    val = SimpleValidation([sample], opts.outputDir[0])
    htmlReport = val.createHtmlReport(validationName=opts.html_validation_name)   

    if opts.collection==2:
    	hgchit = [hgcalPlots.hgcalHitPlotter]
    	val.doPlots(hgchit, plotterDrawArgs=drawArgs)   
    else :
	hgclayclus = [hgcalPlots.hgcalLayerClustersPlotter]
        val.doPlots(hgclayclus, plotterDrawArgs=drawArgs)
	if(opts.collection==3):
		sample = SimpleSample(opts.subdirprefix[1], opts.html_sample, filenames)
		val = SimpleValidation([sample], opts.outputDir[1])
		htmlReport_2 = val.createHtmlReport(validationName=opts.html_validation_name)
		hgchit = [hgcalPlots.hgcalHitPlotter]
        	val.doPlots(hgchit, plotterDrawArgs=drawArgs)

    print()
    if opts.no_html:
        print("Plots created into directory '%s'." % opts.outputDir)
    else:
        htmlReport.write()
	if(opts.collection==3):
		htmlReport_2.write()
        print("Plots and HTML report created into directory '%s'. You can just move it to some www area and access the pages via web browser" % (','.join(opts.outputDir)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create set of HGCal validation plots from one or more DQM files.")
    parser.add_argument("files", metavar="file", type=str, nargs="+",
                        help="DQM file to plot the validation plots from")
    parser.add_argument("-o", "--outputDir", type=str, default=["plots1","plots2"], nargs="+",
                        help="Plot output directories (default: 'plots1'")
    parser.add_argument("--subdirprefix", type=str, default=["plots1","plots2"], nargs="+",
                        help="Prefix for subdirectories inside outputDir (default: 'plots1')")
    parser.add_argument("--no-ratio", action="store_true",
                        help="Disable ratio pads")
    parser.add_argument("--separate", action="store_true",
                        help="Save all plots separately instead of grouping them")
    parser.add_argument("--png", action="store_true",
                        help="Save plots in PNG instead of PDF")
    parser.add_argument("--no-html", action="store_true",
                        help="Disable HTML page generation")
    parser.add_argument("--html-sample", default="Sample",
                        help="Sample name for HTML page generation (default 'Sample')")
    parser.add_argument("--html-validation-name", default="",
                        help="Validation name for HTML page generation (enters to <title> element) (default '')")
    parser.add_argument("--verbose", action="store_true",
                        help="Be verbose")
    parser.add_argument("-c", "--collection", type=int, choices=[1,2,3], default=1,
                        help="Choose plots collection: 1 for layerCluster , 2 for hitValidation , 3 for both")    

    opts = parser.parse_args()

    if opts.collection == 3 and len(opts.outputDir)==1:
	raise RuntimeError("need to assign names for both dirrectories")

    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)

    main(opts)
