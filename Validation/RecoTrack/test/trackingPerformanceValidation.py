#! /usr/bin/env python

from Validation.RecoTrack.plotting.validation import Sample, Validation
import Validation.RecoTrack.plotting.validation as validation
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots

#########################################################
########### User Defined Variables (BEGIN) ##############

### Reference release
RefRelease='CMSSW_7_6_0_pre4'

### Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_7_6_0_pre5'

### This is the list of IDEAL-conditions relvals 
startupsamples= [
    Sample('RelValMinBias', midfix="13"),
    Sample('RelValTTbar', midfix="13"),
    Sample('RelValQCD_Pt_600_800', midfix="13"),
    Sample('RelValQCD_Pt_3000_3500', midfix="13"),
    Sample('RelValQCD_FlatPt_15_3000', append="HS", midfix="13"),
    Sample('RelValZMM', midfix="13"),
    Sample('RelValWjet_Pt_3000_3500', midfix="13"),
    Sample('RelValSingleElectronPt35', midfix="UP15"),
    Sample('RelValSingleElectronPt10', midfix="UP15"),
    Sample('RelValSingleMuPt10', midfix="UP15"),
    Sample('RelValSingleMuPt100', midfix="UP15")
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

upgradesamples = [
#    Sample("RelValTTbar", midfix="14TeV", scenario="UPG2019withGEM"  ),
#    Sample("RelValTTbar", midfix="14TeV", scenario="UPG2023SHNoTaper"),
#    Sample('RelValQCD_Pt_3000_3500', midfix="14TeV", scenario="UPG2019withGEM"  ),
#    Sample('RelValQCD_Pt_3000_3500', midfix="14TeV", scenario="UPG2023SHNoTaper"),
#    Sample('RelValSingleElectronPt35', scenario="UPG2019withGEM"  ),
#    Sample('RelValSingleElectronPt35', scenario="UPG2023SHNoTaper"),
#    Sample('RelValSingleElectronPt10', scenario="UPG2019withGEM"  ),
#    Sample('RelValSingleElectronPt10', scenario="UPG2023SHNoTaper"),
#    Sample('RelValSingleMuPt10', scenario="UPG2019withGEM"  ),
#    Sample('RelValSingleMuPt10', scenario="UPG2023SHNoTaper"),
#    Sample('RelValSingleMuPt100', scenario="UPG2019withGEM"  ),
#    Sample('RelValSingleMuPt100', scenario="UPG2023SHNoTaper"),
#    Sample('RelValTenMuExtendedE_0_200', scenario="UPG2019withGEM"  ),
#    Sample('RelValTenMuExtendedE_0_200', scenario="UPG2023SHNoTaper"),
]

fastsimstartupsamples = [
    Sample('RelValTTbar', midfix="13", fastsim=True),
    Sample('RelValQCD_FlatPt_15_3000', midfix="13", fastsim=True),
    Sample('RelValSingleMuPt10', midfix="UP15", fastsim=True),
    Sample('RelValSingleMuPt100', midfix="UP15", fastsim=True)
]

pileupfastsimstartupsamples = [
    Sample('RelValTTbar', putype=putype("25ns"), midfix="13", fastsim=True)
]

doFastVsFull = True
if "_pmx" in NewRelease:
    startupsamples = []
    fastsimstartupsamples = []
    doFastVsFull = False
    if not NewRelease in validation._globalTags:
        validation._globalTags[NewRelease] = validation._globalTags[NewRelease.replace("_pmx", "")]

### Track algorithm name and quality. Can be a list.
Algos= ['ootb', 'initialStep', 'lowPtTripletStep','pixelPairStep','detachedTripletStep','mixedTripletStep','pixelLessStep','tobTecStep','jetCoreRegionalStep','muonSeededStepInOut','muonSeededStepOutIn',
        'ak4PFJets','btvLike'
]
#Algos= ['ootb']
Qualities=['', 'highPurity']
VertexCollections=["offlinePrimaryVertices", "selectedOfflinePrimaryVertices"]

def limitProcessing(algo, quality):
    return algo in Algos and quality in Qualities

### Reference and new repository
RefRepository = '/afs/cern.ch/cms/Physics/tracking/validation/MC'
NewRepository = 'new' # copy output into a local folder

# Tracking validation plots
val = Validation(
    fullsimSamples = startupsamples + pileupstartupsamples + upgradesamples,
    fastsimSamples = fastsimstartupsamples + pileupfastsimstartupsamples,
    refRelease=RefRelease, refRepository=RefRepository,
    newRelease=NewRelease, newRepository=NewRepository
)
htmlReport = val.createHtmlReport()
val.download()
val.doPlots(plotter=trackingPlots.plotter, plotterDrawArgs={"ratio": True},
#            limitSubFoldersOnlyTo={"": limitProcessing},
            htmlReport=htmlReport, doFastVsFull=doFastVsFull
)

valv = Validation(
    fullsimSamples = pileupstartupsamples, fastsimSamples=[],
    refRelease=RefRelease, refRepository=RefRepository,
    newRelease=NewRelease, newRepository=NewRepository)
valv.download()
valv.doPlots(plotter=vertexPlots.plotter, plotterDrawArgs={"ratio": True},
             limitSubFoldersOnlyTo={"": VertexCollections},
             htmlReport=htmlReport, doFastVsFull=doFastVsFull
)
htmlReport.write()


# Timing plots
#val2 = validation.Validation(
#    fullsimSamples = startupsamples, fastsimSamples=[],
#    newRelease=NewRelease)
#val2.doPlots(refRelease=RefRelease,
#             refRepository=RefRepository, newRepository=NewRepository, plotter=trackingPlots.timePlotter,
#             algos=None, qualities=None)

# TrackingParticle plots
#val3 = validation.Validation(
#    fullsimSamples = startupsamples + pileupstartupsamples + upgradesamples,
#    fastsimSamples=[], newRelease=NewRelease,
#    selectionName="_tp")
#val3.download()
#val3.doPlots(algos=None, qualities=None, refRelease=RefRelease,
#             refRepository=RefRepository, newRepository=NewRepository, plotter=trackingPlots.tpPlotter)

