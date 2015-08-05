#! /usr/bin/env python

from Validation.RecoTrack.plotting.validation import Sample, Validation
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots

#########################################################
########### User Defined Variables (BEGIN) ##############

### Reference release
RefRelease='CMSSW_7_5_0'

### Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_7_6_0_pre1'

### This is the list of IDEAL-conditions relvals 
startupsamples= [
    Sample('RelValMinBias', midfix="13"),
    Sample('RelValTTbar', midfix="13"),
    Sample('RelValQCD_Pt_3000_3500', midfix="13"),
    Sample('RelValQCD_Pt_600_800', midfix="13"),
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
    Sample('RelValTTbar', putype="25ns", midfix="13", fastsim=True)
]

if "_pmx" in NewRelease:
    startupsamples = []
    fastsimstartupsamples = []
    pileupfastsimstartupsamples = []

### Track algorithm name and quality. Can be a list.
Algos= ['ootb', 'initialStep', 'lowPtTripletStep','pixelPairStep','detachedTripletStep','mixedTripletStep','pixelLessStep','tobTecStep','jetCoreRegionalStep','muonSeededStepInOut','muonSeededStepOutIn',
        'ak4PFJets','btvLike'
]
#Algos= ['ootb']
Qualities=['', 'highPurity']

def limitProcessing(algo, quality):
    return algo in Algos and quality in Qualities

### Reference and new repository
RefRepository = '/afs/cern.ch/cms/Physics/tracking/validation/MC'
NewRepository = 'new' # copy output into a local folder

# Tracking validation plots
val = Validation(
    fullsimSamples = startupsamples + pileupstartupsamples + upgradesamples,
    fastsimSamples = fastsimstartupsamples + pileupfastsimstartupsamples,
    newRelease=NewRelease,
)
val.download()
val.doPlots(refRelease=RefRelease,
            refRepository=RefRepository, newRepository=NewRepository, plotter=trackingPlots.plotter,
            plotterDrawArgs={"ratio": True},
            limitSubFoldersOnlyTo={"": limitProcessing}
)

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

