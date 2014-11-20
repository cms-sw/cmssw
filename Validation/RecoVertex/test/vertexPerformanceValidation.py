#! /usr/bin/env python

from Validation.RecoTrack.plotting.validation import Sample
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots

#########################################################
########### User Defined Variables (BEGIN) ##############

#import Validation.RecoTrack.plotting.plotting as plotting
#plotting.missingOk = True

### Reference release
RefRelease='CMSSW_7_3_0_pre1'

### Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_7_3_0_pre2'

### This is the list of IDEAL-conditions relvals 
startupsamples= [
#    Sample('RelValMinBias', midfix="13"),
#    Sample('RelValTTbar', midfix="13"),
#    Sample('RelValQCD_Pt_3000_3500', midfix="13"),
#    Sample('RelValQCD_Pt_600_800', midfix="13"),
#    Sample('RelValSingleElectronPt35', midfix="UP15"),
#    Sample('RelValSingleElectronPt10', midfix="UP15"),
#    Sample('RelValSingleMuPt10', midfix="UP15"),
#    Sample('RelValSingleMuPt100', midfix="UP15")
]

pileupstartupsamples = [
    Sample('RelValTTbar', putype="25ns", midfix="13"),
    Sample('RelValTTbar', putype="50ns", midfix="13")
]

### Vertex collections
Collections = ["offlinePrimaryVertices", "selectedOfflinePrimaryVertices"]
Qualities=None

### Reference and new repository
RefRepository = '/afs/cern.ch/cms/Physics/tracking/validation/MC'
NewRepository = 'new' # copy output into a local folder


validation = vertexPlots.VertexValidation(
    fullsimSamples = startupsamples + pileupstartupsamples,
    fastsimSamples=[], newRelease=NewRelease)
validation.download()
validation.doPlots(algos=Collections, qualities=Qualities, refRelease=RefRelease,
                   refRepository=RefRepository, newRepository=NewRepository, plotter=vertexPlots.plotter)


