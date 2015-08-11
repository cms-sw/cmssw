#!/usr/bin/env python

# This is an example of plotting the standard tracking validation
# plots from an explicit set of DQM root files.

from Validation.RecoTrack.plotting.validation import SimpleValidation
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots


# Example of file - label pairs
filesLabels = [
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_1.root", "Option 1"),
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_2.root", "Option 2"),
]

outputDir = "plots"
subdirprefix = "sample"

val = SimpleValidation([x[0] for x in filesLabels], [x[1] for x in filesLabels], outputDir)
val.doPlots(trackingPlots.plotter, subdirprefix=subdirprefix, plotterDrawArgs={"ratio": True})

