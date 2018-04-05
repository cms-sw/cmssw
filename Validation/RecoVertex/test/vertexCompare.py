#!/usr/bin/env python

# This is an example of plotting the standard vertex validation
# plots from an explicit set of DQM root files.

import Validation.RecoTrack.plotting.plotting as plotting
from Validation.RecoTrack.plotting.validation import SimpleValidation
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots


# Example of file - label pairs
filesLabels = [
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_1.root", "Option 1"),
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_2.root", "Option 2"),
]

outputDir = "plots"

### Track algorithm name and quality. Can be a list.
Collections = ["offlinePrimaryVertices", "selectedOfflinePrimaryVertices"]
Qualities=None

def newdirname(algo, quality):
    ret = ""
    if algo is not None:
        ret += "_"+algo
    return ret


val = SimpleValidation([x[0] for x in filesLabels], [x[1] for x in filesLabels], outputDir)
val.doPlots(Collections, Qualities, vertexPlots.plotter, algoDirMap=lambda a, q: a, newdirFunc=newdirname)
