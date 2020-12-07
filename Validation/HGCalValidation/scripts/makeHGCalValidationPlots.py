#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
from time import time

from Validation.RecoTrack.plotting.validation import SeparateValidation, SimpleValidation, SimpleSample
import Validation.HGCalValidation.hgcalPlots as hgcalPlots
import Validation.RecoTrack.plotting.plotting as plotting

trackstersIters = ["ticlMultiClustersFromTrackstersMerge", "ticlMultiClustersFromTrackstersMIP",
                   "ticlMultiClustersFromTrackstersTrk","ticlMultiClustersFromTrackstersTrkEM",
                   "ticlMultiClustersFromTrackstersEM", "ticlMultiClustersFromTrackstersHAD",
                   "ticlMultiClustersFromTrackstersDummy"]

layerClustersGeneralLabel = 'hgcalLayerClusters'
multiclustersGeneralLabel = 'hgcalMultiClusters'
trackstersGeneralLabel = 'allTiclMultiClusters'
hitValidationLabel = 'hitValidation'
hitCalibrationLabel = 'hitCalibration'
caloParticlesLabel = 'caloParticles'
allLabel = 'all'

collection_choices = [layerClustersGeneralLabel]
collection_choices.extend([multiclustersGeneralLabel]+[trackstersGeneralLabel]+[hitValidationLabel]+[hitCalibrationLabel]+[allLabel]+[caloParticlesLabel])

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
    if opts.separate:
        val = SeparateValidation([sample], opts.outputDir[0])
    htmlReport = val.createHtmlReport(validationName=opts.html_validation_name[0])   

    if opts.collection==layerClustersGeneralLabel:
	hgclayclus = [hgcalPlots.hgcalLayerClustersPlotter]
	hgcalPlots.append_hgcalLayerClustersPlots("hgcalLayerClusters", "Layer Clusters")
        val.doPlots(hgclayclus, plotterDrawArgs=drawArgs)
    elif opts.collection == multiclustersGeneralLabel:
        hgcmulticlus = [hgcalPlots.hgcalMultiClustersPlotter]
        hgcalPlots.append_hgcalMultiClustersPlots(multiclustersGeneralLabel, "MultiClusters")
        val.doPlots(hgcmulticlus, plotterDrawArgs=drawArgs)
    elif (opts.collection == trackstersGeneralLabel) :
        hgcmulticlus = [hgcalPlots.hgcalMultiClustersPlotter]
        for i_iter in trackstersIters :
            tracksterCollection = i_iter.replace("ticlMultiClustersFromTracksters","ticlTracksters")
            hgcalPlots.append_hgcalMultiClustersPlots(i_iter, tracksterCollection)
        val.doPlots(hgcmulticlus, plotterDrawArgs=drawArgs)
    elif opts.collection==caloParticlesLabel:
        particletypes = {"pion-":"-211", "pion+":"211", "pion0": "111",
                         "muon-": "-13", "muon+":"13", 
                         "electron-": "-11", "electron+": "11", "photon": "22", 
                         "kaon-": "-321", "kaon+": "321"}
        hgcaloPart = [hgcalPlots.hgcalCaloParticlesPlotter]
        for i_part, i_partID in particletypes.iteritems() :
            hgcalPlots.append_hgcalCaloParticlesPlots(sample.files(), i_partID, i_part)
        val.doPlots(hgcaloPart, plotterDrawArgs=drawArgs)
    elif opts.collection==hitValidationLabel:
    	hgchit = [hgcalPlots.hgcalHitPlotter]
        hgcalPlots.append_hgcalHitsPlots('HGCalSimHitsV', "Simulated Hits")
        hgcalPlots.append_hgcalHitsPlots('HGCalRecHitsV', "Reconstruced Hits")
        hgcalPlots.append_hgcalDigisPlots('HGCalDigisV', "Digis")
    	val.doPlots(hgchit, plotterDrawArgs=drawArgs)   
    elif opts.collection==hitCalibrationLabel:
        hgchitcalib = [hgcalPlots.hgcalHitCalibPlotter]
        val.doPlots(hgchitcalib, plotterDrawArgs=drawArgs)
    else :
        #caloparticles
        particletypes = {"pion-":"-211", "pion+":"211", "pion0": "111",
                         "muon-": "-13", "muon+":"13", 
                         "electron-": "-11", "electron+": "11", "photon": "22", 
                         "kaon-": "-321", "kaon+": "321"}
        hgcaloPart = [hgcalPlots.hgcalCaloParticlesPlotter]
        for i_part, i_partID in particletypes.iteritems() :
            hgcalPlots.append_hgcalCaloParticlesPlots(sample.files(), i_partID, i_part)
        val.doPlots(hgcaloPart, plotterDrawArgs=drawArgs)

        #hits
    	hgchit = [hgcalPlots.hgcalHitPlotter]
        hgcalPlots.append_hgcalHitsPlots('HGCalSimHitsV', "Simulated Hits")
        hgcalPlots.append_hgcalHitsPlots('HGCalRecHitsV', "Reconstruced Hits")
        hgcalPlots.append_hgcalDigisPlots('HGCalDigisV', "Digis")
    	val.doPlots(hgchit, plotterDrawArgs=drawArgs)   

        #calib
        hgchitcalib = [hgcalPlots.hgcalHitCalibPlotter]
        val.doPlots(hgchitcalib, plotterDrawArgs=drawArgs)

 	#layer clusters 
	hgclayclus = [hgcalPlots.hgcalLayerClustersPlotter]
	hgcalPlots.append_hgcalLayerClustersPlots("hgcalLayerClusters", "Layer Clusters")
	val.doPlots(hgclayclus, plotterDrawArgs=drawArgs)

        #multiclusters
        hgcmulticlus = [hgcalPlots.hgcalMultiClustersPlotter]
        for i_iter in trackstersIters :
            tracksterCollection = i_iter.replace("ticlMultiClustersFromTracksters","ticlTracksters")
            hgcalPlots.append_hgcalMultiClustersPlots(i_iter, tracksterCollection)
        val.doPlots(hgcmulticlus, plotterDrawArgs=drawArgs)


    if opts.no_html:
        print("Plots created into directory '%s'." % opts.outputDir)
    else:
        htmlReport.write()

        print("Plots and HTML report created into directory '%s'. You can just move it to some www area and access the pages via web browser" % (','.join(opts.outputDir)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create set of HGCal validation plots from one or more DQM files.")
    parser.add_argument("files", metavar="file", type=str, nargs="+", 
                        default = "DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root",
                        help="DQM file to plot the validation plots from")
    parser.add_argument("-o", "--outputDir", type=str, default=["plots1","plots2"], nargs="+",
                        help="Plot output directories (default: 'plots1'")
    parser.add_argument("--subdirprefix", type=str, default=["plots1","plots2"], nargs="+",
                        help="Prefix for subdirectories inside outputDir (default: 'plots1')")
    parser.add_argument("--no-ratio", action="store_true", default = False,
                        help="Disable ratio pads")
    parser.add_argument("--separate", action="store_true", default = False,
                        help="Save all plots separately instead of grouping them")
    parser.add_argument("--png", action="store_true",
                        help="Save plots in PNG instead of PDF")
    parser.add_argument("--no-html", action="store_true", default = False,
                        help="Disable HTML page generation")
    parser.add_argument("--html-sample", default="Sample",
                        help="Sample name for HTML page generation (default 'Sample')")
    parser.add_argument("--html-validation-name", type=str, default=["",""], nargs="+",
                        help="Validation name for HTML page generation (enters to <title> element) (default '')")
    parser.add_argument("--verbose", action="store_true", default = False,
                        help="Be verbose")
    parser.add_argument("--collection", choices=collection_choices, default=layerClustersGeneralLabel,
                        help="Choose output plots collections among possible choices")    

    opts = parser.parse_args()

    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)

    main(opts)
