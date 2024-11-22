#!/usr/bin/env python3

import os
import argparse
import datetime

from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels, ticlIterLabels_v5

from Validation.RecoTrack.plotting.validation import SeparateValidation, SimpleValidation, SimpleSample
from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator
import Validation.HGCalValidation.hgcalPlots as hgcalPlots
import Validation.RecoTrack.plotting.plotting as plotting

simClustersIters = [hgcalValidator.label_SimClustersLevel._InputTag__moduleLabel, "ticlSimTracksters"]

hitCalLabel = 'hitCalibration'
hitValLabel = 'hitValidation'
layerClustersLabel = 'layerClusters'
trackstersLabel = 'tracksters'
trackstersWithEdgesLabel = 'trackstersWithEdges'
candidatesLabel = 'candidates'
simLabel = 'simulation'
allLabel = 'all'
ticlVersions = [4, 5]
ticlVersion = 4
collection_choices = [allLabel]
collection_choices.extend([hitCalLabel]+[hitValLabel]+[layerClustersLabel]+[trackstersLabel]+[trackstersWithEdgesLabel]+[candidatesLabel]+[simLabel])
tracksters = []

def main(opts):

    drawArgs={}
    extendedFlag = False
    if opts.no_ratio:
        drawArgs["ratio"] = False
    if opts.separate:
        drawArgs["separate"] = True
    if opts.png:
        drawArgs["saveFormat"] = ".png"
    if opts.extended:
        extendedFlag = True
    if opts.verbose:
        plotting.verbose = True
    if opts.ticlv == 5:
        ticlVersion = 5

    filenames = [(f, f.replace(".root", "")) for f in opts.files]
    sample = SimpleSample(opts.subdirprefix[0], opts.html_sample, filenames)

    val = SimpleValidation([sample], opts.outputDir[0], nProc=opts.jobs)
    if opts.separate:
        val = SeparateValidation([sample], opts.outputDir[0])
    htmlReport = val.createHtmlReport(validationName=opts.html_validation_name[0])
    trackstersIters = []
    if (ticlVersion == 5):
        trackstersIters = ticlIterLabels_v5.copy()
        trackstersIters.extend(['ticlTracksterLinksSuperclusteringDNN','ticlTracksterLinksSuperclusteringMustache'])
    else:
        trackstersIters = ticlIterLabels.copy()
    trackstersIters.extend(['ticlSimTracksters', 'ticlSimTracksters_fromCPs'])
    #layerClusters
    def plot_LC():
        hgclayclus = [hgcalPlots.hgcalLayerClustersPlotter]
        hgcalPlots.append_hgcalLayerClustersPlots(hgcalValidator.label_layerClusterPlots._InputTag__moduleLabel, "Layer Clusters", extendedFlag)
        val.doPlots(hgclayclus, plotterDrawArgs=drawArgs)

    #simClusters
    def plot_SC():
        hgcsimclus = [hgcalPlots.hgcalSimClustersPlotter]
        for i_iter in simClustersIters:
            hgcalPlots.append_hgcalSimClustersPlots(i_iter, i_iter)
        val.doPlots(hgcsimclus, plotterDrawArgs=drawArgs)

    #tracksters
    def plot_Tst():

        hgctrackster = [hgcalPlots.hgcalTrackstersPlotter]
        for tracksterCollection in trackstersIters :
            print("Searching for tracksters collection in DQM files: ", tracksterCollection)

            hgcalPlots.append_hgcalTrackstersPlots(tracksterCollection, tracksterCollection)
        val.doPlots(hgctrackster, plotterDrawArgs=drawArgs)

    #trackstersWithEdges
    def plot_TstEdges():
        plot_Tst()
        for tracksterCollection in trackstersIters :
            hgctracksters = [hgcalPlots.create_hgcalTrackstersPlotter(sample.files(), tracksterCollection, tracksterCollection)]
            val.doPlots(hgctracksters, plotterDrawArgs=drawArgs)

    #caloParticles
    def plot_CP():
        particletypes = {"pion-":"-211", "pion+":"211", "pion0": "111",
                         "muon-": "-13", "muon+":"13",
                         "electron-": "-11", "electron+": "11", "photon": "22",
                         "kaon0L": "310", "kaon0S": "130",
                         "kaon-": "-321", "kaon+": "321"}
        hgcaloPart = [hgcalPlots.hgcalCaloParticlesPlotter]
        for i_part, i_partID in particletypes.items() :
            hgcalPlots.append_hgcalCaloParticlesPlots(sample.files(), i_partID, i_part)
        val.doPlots(hgcaloPart, plotterDrawArgs=drawArgs)

    #hitValidation
    def plot_hitVal():
        hgchit = [hgcalPlots.hgcalHitPlotter]
        hgcalPlots.append_hgcalHitsPlots('HGCalSimHitsV', "Simulated Hits")
        hgcalPlots.append_hgcalHitsPlots('HGCalRecHitsV', "Reconstruced Hits")
        hgcalPlots.append_hgcalDigisPlots('HGCalDigisV', "Digis")
        val.doPlots(hgchit, plotterDrawArgs=drawArgs)

    #hitCalibration
    def plot_hitCal():
        hgchitcalib = [hgcalPlots.hgcalHitCalibPlotter]
        val.doPlots(hgchitcalib, plotterDrawArgs=drawArgs)

    def plotCand():
        ticlcand = [hgcalPlots.hgcalTICLCandPlotter]
        val.doPlots(ticlcand, plotterDrawArgs=drawArgs)

    plotDict = {hitCalLabel:[plot_hitCal], hitValLabel:[plot_hitVal], layerClustersLabel:[plot_LC], trackstersLabel:[plot_Tst], trackstersWithEdgesLabel:[plot_TstEdges], simLabel:[plot_SC, plot_CP], candidatesLabel:[plotCand]}

    if (opts.collection != allLabel):
        for task in plotDict[opts.collection]:
            task()
    else:
        for label in plotDict:
            if (label == trackstersLabel): continue # already run in trackstersWithEdges
            for task in plotDict[label]:
                task()

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
    parser.add_argument("--png", action="store_true", default = True,
                        help="Save plots in PNG instead of PDF")
    parser.add_argument("--no-html", action="store_true", default = False,
                        help="Disable HTML page generation")
    parser.add_argument("--html-sample", default=os.environ['CMSSW_VERSION'],
                        help="Sample name for HTML page generation (default: CMSSW version)")
    parser.add_argument("--html-validation-name", type=str, default=["TICL Validation",""], nargs="+",
                        help="Validation name for HTML page generation (enters to <title> element) (default 'TICL Validation')")
    parser.add_argument("--collection", choices=collection_choices, default=layerClustersLabel,
                        help="Choose output plots collections among possible choices")
    parser.add_argument("--extended", action="store_true", default = False,
                        help="Include extended set of plots (e.g. bunch of distributions; default off)")
    parser.add_argument("--jobs", default=0, type=int,
                        help="Number of jobs to run in parallel for generating plots. Default is 0 i.e. run number of cpu cores jobs.")
    parser.add_argument("--ticlv", choices=ticlVersions, default=4, type=int,
                        help="TICL Version. Specify 4 or 5. Default 4.")
    parser.add_argument("--verbose", action="store_true", default = False,
                        help="Be verbose")

    opts = parser.parse_args()

    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)

    main(opts)
