#!/usr/bin/env python3

# This is an example of plotting the standard tracking validation
# plots from an explicit set of DQM root files.

from Validation.RecoTrack.plotting.validation import SimpleValidation, SimpleSample
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots

# Below is an example on how to make plots for custom
# track+TrackingParticle selections (e.g. selecting a specific eta-phi
# region). Track selection is handled by defining custom track
# "quality" (string in the track collection names), TrackingParticle
# selection by MTV instances having the same string as their postfix.
# See python/customiseMTVForBPix123Holes.py for a customise function
# setting up the MTV instances for CMSSW job.
#
#trackingPlots._additionalTrackQualities.extend(["L1L2", "L2L3"])
#for pfix in ["L1L2", "L2L3"]:
#    trackingPlots._appendTrackingPlots("Track"+pfix, "", trackingPlots._simBasedPlots+trackingPlots._recoBasedPlots)
#    trackingPlots._appendTrackingPlots("TrackSeeding"+pfix, "", trackingPlots._seedingBuildingPlots, seeding=True)
#    trackingPlots._appendTrackingPlots("TrackBuilding"+pfix, "", trackingPlots._seedingBuildingPlots)

outputDir = "plots" # Plot output directory
description = "Short description of your comparison"

plotterDrawArgs = dict(
    separate=False, # Set to true if you want each plot in it's own canvas
#    ratio=False,   # Uncomment to disable ratio pad
)

# Pairs of file names and legend labels
filesLabels = [
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_1.root", "Option 1"),
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_2.root", "Option 2"),
]
# Files are grouped together as a "sample" (the files don't
# necessarily have to come from the same sample, like ttbar, but this
# is the abstraction here)
sample = SimpleSample("sample_prefix", # Prefix for subdirectory names
                      "Sample name",   # The name appears in the HTML pages
                      filesLabels)     # Files and legend labels

# You can produce plots for multiple samples on one. Just construct
# multiple SimpleSample objects like above and add them to the list
# below.
samples = [
    sample
]

# Example of how to limit tracking plots to specific iterations
kwargs_tracking = {}
class LimitTrackAlgo: # helper class to limit to iterations
    def __init__(self, algos):
        self._algos = algos
    def __call__(self, algo, quality):
        if self._algos is not None:
            if algo not in self._algos:
                return False
        if "Pt09" in quality:
            return False
        if "ByAlgoMask" in quality or "ByOriginalAlgo" in quality:
            return False
        return True
limit = LimitTrackAlgo(["ootb", "initialStep"]) # limit to generalTracks (ootb) and initialStep
ignore = lambda algo, quality: False # ignore everything
ignore09 = LimitTrackAlgo(None) # ignore Pt09 plots

# This specifies how different sets of plots are treated. If some
# "plot set" is not in the dictionary, full set of plots will be
# produced for it
limitSubFolders = {
    "":                limit,  # The default set (signal TrackingParticles for efficiency, all TrackingParticles for fakes)
    "tpPtLess09":      limit,  # Efficiency for TrackingParticles with pT < 0.9 GeV
    "tpEtaGreater2p7": limit,  # Efficiency for TrackingParticles with |eta| > 2.7 (phase 2)
    "allTPEffic":      ignore, # Efficiency with all TrackingParticles
    "bhadron":         limit,  # Efficiency with B-hadron TrackingParticles
    "displaced":       limit,  # Efficiency for TrackingParticles with no tip or lip cuts
    "fromPV":          limit,  # Tracks from PV, signal TrackingParticles for efficiency and fakes
    "fromPVAllTP":     limit,  # Tracks from PV, all TrackingParticles for fakes
    "building":        ignore, # Built tracks (as opposed to selected tracks in above)
    "seeding":         ignore, # Seeds
}
# arguments to be passed to tracking val.doPlots() below
kwargs_tracking["limitSubFoldersOnlyTo"]=limitSubFolders

# Example of how to customize the plots, here applied only if each
# plot is drawn separately
if plotterDrawArgs["separate"]:
    common = dict(
        title=""
    )

    for plotFolderName in ["", "building"]: # these refer to the various cases added with _appendTrackingPlots in trackingPlots.py
        # Get the PlotFolder object
        plotFolder = trackingPlots.plotter.getPlotFolder(plotFolderName)

        # These are the PlotGroup objects defined in trackingPlots.py,
        # name is the same as the first parameter to PlotGroup constructor
        plotGroup = plotFolder.getPlotGroup("effandfake1")
        # From PlotGroup one can ask an individual Plot, again name is
        # the same as used for Plot constructor. The setProperties()
        # accepts the same parameters as the constructor, see
        # plotting.Plot for more information.
        plotGroup.getPlot("efficPt").setProperties(legendDx=-0, legendDy=-0, **common)

    # Example of customization of vertex plots
    common["lineWidth"] = 4
    plotFolder = vertexPlots.plotterExt.getPlotFolder("gen")
    plotGroup = plotFolder.getPlotGroup("genpos")
    plotGroup.getPlot("GenAllV_Z").setProperties(xtitle="Simulated vertex z (cm)", legendDy=-0.1, legendDx=-0.45, ratioYmax=2.5, **common)


val = SimpleValidation(samples, outputDir)
report = val.createHtmlReport(validationName=description)
val.doPlots([
    trackingPlots.plotter,     # standard tracking plots
    #trackingPlots.plotterExt, # extended tracking plots (e.g. distributions)
],
            plotterDrawArgs=plotterDrawArgs,
            **kwargs_tracking
)
val.doPlots([
    #trackingPlots.timePlotter, # tracking timing plots
    vertexPlots.plotter,        # standard vertex plots
    #vertexPlots.plotterExt,    # extended vertex plots (e.g. distributions)
],
            plotterDrawArgs=plotterDrawArgs,
)
report.write() # comment this if you don't want HTML page generation
