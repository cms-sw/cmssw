#! /usr/bin/env python

from Validation.RecoTrack.plotting.validation import Sample
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots
import Validation.RecoTrack.plotting.validation as validation

#########################################################
########### User Defined Variables (BEGIN) ##############

### Reference release
RefRelease='CMSSW_7_2_0_pre8'

### Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_7_3_0_pre1'

#import Validation.RecoTrack.plotting.plotting as plotting
#plotting.missingOk = True

### This is the list of IDEAL-conditions relvals 
startupsamples= [
    Sample('RelValMinBias', midfix="13"),
    Sample('RelValTTbar', midfix="13"),
    Sample('RelValQCD_Pt_3000_3500', midfix="13"),
    Sample('RelValQCD_Pt_600_800', midfix="13"),
    Sample('RelValSingleElectronPt35', midfix="UP15"),
    Sample('RelValSingleElectronPt10', midfix="UP15"),
    Sample('RelValSingleMuPt10', midfix="UP15"),
    Sample('RelValSingleMuPt100', midfix="UP15")
]

pileupstartupsamples = [
    Sample('RelValTTbar', putype="25ns", midfix="13"),
    Sample('RelValTTbar', putype="50ns", midfix="13")
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
    Sample('RelValTTbar', midfix="13", fastsim=True)
]

pileupfastsimstartupsamples = [
    Sample('RelValTTbar', putype="AVE20", midfix="13", fastsim=True, fastsimCorrespondingFullsimPileup="50ns")
]

### Track algorithm name and quality. Can be a list.
Algos= ['ootb', 'iter0', 'iter1','iter2','iter3','iter4','iter5','iter6','iter7','iter9','iter10']
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
