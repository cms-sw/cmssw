#!/usr/bin/env python

# This is an example of plotting the standard tracking validation
# plots from an explicit set of DQM root files.

import Validation.RecoTrack.plotting.plotting as plotting
from Validation.RecoTrack.plotting.validation import SimpleValidation
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots



# Example of file - label pairs
filesLabels = [
    ("DQM_CMSSW_7_6_X_2015-07-31-2300_AfterSeedStateBackPropagation.root", "Changed_FastSim_7_6_X_AfterSeedStateBackPropagation"),
    ("DQM_V0001_R000000001__RelValTTbar_13__CMSSW_7_5_0_pre3-MCRUN2_74_V7_FastSim-v1__DQMIO.root", "StandardFastSim_7_5_0_pre3"),
]

outputDir = "plots"

### Track algorithm name and quality. Can be a list.
Algos= ['ootb', 'initialStep', 'lowPtTripletStep','pixelPairStep','detachedTripletStep','mixedTripletStep','pixelLessStep','tobTecStep','jetCoreRegionalStep','muonSeededStepInOut','muonSeededStepOutIn']
#Algos= ['ootb']
Qualities=['', 'highPurity']

def newdirname(algo, quality):
    ret = ""
    if quality != "":
        ret += "_"+quality
    if not (algo == "ootb" and quality != ""):
        ret += "_"+algo

    return ret


val = SimpleValidation([x[0] for x in filesLabels], [x[1] for x in filesLabels], outputDir)
val.doPlots(Algos, Qualities, trackingPlots.plotter, algoDirMap=trackingPlots._tracks_map, newdirFunc=newdirname)

