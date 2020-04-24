#! /usr/bin/env python

from Validation.RecoTrack.plotting.validation import Sample, Validation
import Validation.RecoTrack.plotting.validation as validation
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots

#########################################################
########### User Defined Variables (BEGIN) ##############

### Reference release
RefRelease='CMSSW_7_5_0'

### Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_7_6_0_pre1'

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

def putype(t):
    if "_pmx" in NewRelease:
        return {"default": t, NewRelease: "pmx"+t}
    return t

pileupstartupsamples = [
    Sample('RelValTTbar', putype=putype("25ns"), midfix="13"),
    Sample('RelValTTbar', putype=putype("50ns"), midfix="13"),
    Sample('RelValZMM', putype=putype("25ns"), midfix="13"),
    Sample('RelValZMM', putype=putype("50ns"), midfix="13")
]

if "_pmx" in NewRelease:
    if not NewRelease in validation._globalTags:
        validation._globalTags[NewRelease] = validation._globalTags[NewRelease.replace("_pmx", "")]

### Vertex collections
Collections = ["offlinePrimaryVertices", "selectedOfflinePrimaryVertices"]

### Reference and new repository
RefRepository = '/afs/cern.ch/cms/Physics/tracking/validation/MC'
NewRepository = 'new' # copy output into a local folder


validation = Validation(
    fullsimSamples = pileupstartupsamples, fastsimSamples=[],
    refRelease=RefRelease, refRepository=RefRepository,
    newRelease=NewRelease, newRepository=NewRepository)
validation.download()
validation.doPlots(plotter=vertexPlots.plotter, plotterDrawArgs={"ratio": True},
                   limitSubFoldersOnlyTo={"": Collections},
)

#validation2 = vertexPlots.VertexValidation(
#    fullsimSamples = startupsamples + pileupstartupsamples,
#    fastsimSamples=[], newRelease=NewRelease)
#validation2.download()
#validation2.doPlots(algos=None, qualities=Qualities, refRelease=RefRelease,
#                    refRepository=RefRepository, newRepository=NewRepository, plotter=vertexPlots.plotterGen)

