#! /usr/bin/env python

from Validation.RecoTrack.plotting.validation import Sample
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots
import Validation.RecoTrack.plotting.validation as validation

#########################################################
########### User Defined Variables (BEGIN) ##############

### Reference release
RefRelease='CMSSW_7_4_0_pre6'

### Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_7_4_0_pre8'

#import Validation.RecoTrack.plotting.plotting as plotting
#plotting.missingOk = True

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

pileupstartupsamples = [
    Sample('RelValTTbar', putype="25ns", midfix="13"),
    Sample('RelValTTbar', putype="50ns", midfix="13"),
    Sample('RelValZMM', putype="25ns", midfix="13"),
    Sample('RelValZMM', putype="50ns", midfix="13")
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
#    Sample('RelValTTbar', putype="AVE20", midfix="13", fastsim=True, fastsimCorrespondingFullsimPileup="50ns")
]

### Track algorithm name and quality. Can be a list.
Algos= ['ootb', 'initialStep', 'lowPtTripletStep','pixelPairStep','detachedTripletStep','mixedTripletStep','pixelLessStep','tobTecStep','jetCoreRegionalStep','muonSeededStepInOut','muonSeededStepOutIn']
#Algos= ['ootb']
Qualities=['', 'highPurity']

### Reference and new repository
RefRepository = '/afs/cern.ch/cms/Physics/tracking/validation/MC'
NewRepository = 'new' # copy output into a local folder

# Tracking validation plots
val = trackingPlots.TrackingValidation(
    fullsimSamples = startupsamples + pileupstartupsamples + upgradesamples,
    fastsimSamples = fastsimstartupsamples + pileupfastsimstartupsamples,
    newRelease=NewRelease,
)
val.download()
val.doPlots(algos=Algos, qualities=Qualities, refRelease=RefRelease,
                   refRepository=RefRepository, newRepository=NewRepository, plotter=trackingPlots.plotter)

# Timing plots
#val2 = validation.Validation(
#    fullsimSamples = startupsamples, fastsimSamples=[],
#    newRelease=NewRelease)
#val2.doPlots(refRelease=RefRelease,
#             refRepository=RefRepository, newRepository=NewRepository, plotter=trackingPlots.timePlotter,
#             algos=None, qualities=None)

val3 = validation.Validation(
    fullsimSamples = startupsamples + pileupstartupsamples + upgradesamples,
    fastsimSamples=[], newRelease=NewRelease,
    selectionName="_tp")
val3.download()
val3.doPlots(algos=None, qualities=None, refRelease=RefRelease,
             refRepository=RefRepository, newRepository=NewRepository, plotter=trackingPlots.tpPlotter)

