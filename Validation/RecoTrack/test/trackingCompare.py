#!/usr/bin/env python

# This is an example of plotting the standard tracking validation
# plots from an explicit set of DQM root files.

import Validation.RecoTrack.plotting.plotting as plotting
from Validation.RecoTrack.plotting.validation import SimpleValidation
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots



# Example of file - label pairs
filesLabels = [
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_1.root", "Option 1"),
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_2.root", "Option 2"),
]

outputDir = "plots"

### Track algorithm name and quality. Can be a list.
Algos= ['ootb', 'initialStep', 'lowPtTripletStep','pixelPairStep','detachedTripletStep','mixedTripletStep','pixelLessStep','tobTecStep','jetCoreRegionalStep','muonSeededStepInOut','muonSeededStepOutIn']
#Algos= ['ootb']
Qualities=['', 'highPurity']

val = SimpleValidation([x[0] for x in filesLabels], [x[1] for x in filesLabels], outputDir)
val.doPlots(Algos, Qualities, trackingPlots.plotter, algoDirMap=trackingPlots._tracks_map)

