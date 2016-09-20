#! /usr/bin/env python

from Validation.RecoTrack.plotting.validation import Sample, Validation
import Validation.RecoTrack.plotting.validation as validation
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots

#########################################################
########### User Defined Variables (BEGIN) ##############

### Reference release
RefRelease='CMSSW_8_1_0_pre8'

### Relval release (set if different from $CMSSW_VERSION)
NewRelease='CMSSW_8_1_0_pre9'

### This is the list of IDEAL-conditions relvals 
startupsamples= [
    Sample('RelValMinBias', midfix="13"),
    Sample('RelValTTbar', midfix="13"),
    Sample('RelValQCD_Pt_600_800', midfix="13"),
    Sample('RelValQCD_Pt_3000_3500', midfix="13"),
    Sample('RelValQCD_FlatPt_15_3000', append="HS", midfix="13"),
    Sample('RelValZMM', midfix="13"),
    Sample('RelValWjet_Pt_3000_3500', midfix="13"),
    Sample('RelValH125GGgluonfusion', midfix="13"),
    Sample('RelValSingleElectronPt35', midfix="UP15"),
    Sample('RelValSingleElectronPt10', midfix="UP15"),
    Sample('RelValSingleMuPt10', midfix="UP15"),
    Sample('RelValSingleMuPt100', midfix="UP15")
]

def putype(t):
    if "_pmx" in NewRelease:
        if "_pmx" in RefRelease:
            return {"default": "pmx"+t}
        return {"default": t, NewRelease: "pmx"+t}
    return t

pileupstartupsamples = [
    Sample('RelValTTbar', putype=putype("25ns"), punum=35, midfix="13"),
#    Sample('RelValTTbar', putype=putype("50ns"), punum=35, midfix="13"),
    Sample('RelValZMM', putype=putype("25ns"), punum=35, midfix="13"),
#    Sample('RelValZMM', putype=putype("50ns"), punum=35, midfix="13")
]

phase1samples = [
    Sample('RelValMinBias', midfix="13"),
    Sample("RelValTTbar", midfix="13"),
    Sample('RelValZMM', midfix="13"),
    Sample('RelValQCD_Pt_600_800', midfix="13"),
    Sample('RelValTenMuE_0_200'),
    Sample("RelValTTbar", midfix="13", putype=putype("25ns"), punum=35),
    Sample("RelValZMM", midfix="13", putype=putype("25ns"), punum=35),
]

phase2samples = [
    Sample("RelValMinBias", midfix="TuneZ2star_14TeV", scenario="2023GReco"),
    Sample("RelValMinBias", midfix="TuneZ2star_14TeV", scenario="2023tilted"),
    Sample("RelValTTbar", midfix="14TeV", scenario="2023GReco"),
    Sample("RelValTTbar", midfix="14TeV", scenario="2023GRecoPU35", putype=putype("25ns"), punum=35),
    Sample("RelValTTbar", midfix="14TeV", scenario="2023GRecoPU140", putype=putype("25ns"), punum=140),
    Sample("RelValTTbar", midfix="14TeV", scenario="2023GRecoPU200", putype=putype("25ns"), punum=200),
    Sample("RelValTTbar", midfix="14TeV", scenario="2023tilted"),
    Sample("RelValTTbar", midfix="14TeV", scenario="2023tiltedPU35", putype=putype("25ns"), punum=35),
    Sample("RelValTTbar", midfix="14TeV", scenario="2023tiltedPU140", putype=putype("25ns"), punum=140),
    Sample("RelValTTbar", midfix="14TeV", scenario="2023tiltedPU200", putype=putype("25ns"), punum=200),
    Sample("RelValZMM", midfix="13", scenario="2023GReco"),
    Sample("RelValZMM", midfix="13", scenario="2023GRecoPU140", putype=putype("25ns"), punum=140),
    Sample("RelValZMM", midfix="13", scenario="2023GRecoPU200", putype=putype("25ns"), punum=200),
    Sample("RelValZMM", midfix="13", scenario="2023tilted"),
    Sample("RelValZMM", midfix="13", scenario="2023tiltedPU140", putype=putype("25ns"), punum=140),
    Sample("RelValZMM", midfix="13", scenario="2023tiltedPU200", putype=putype("25ns"), punum=200),
    Sample("RelValSingleElectronPt35Extended", scenario="2023GReco"),
    Sample("RelValSingleElectronPt35Extended", scenario="2023tilted"),
    Sample("RelValSingleMuPt10Extended", scenario="2023GReco"),
    Sample("RelValSingleMuPt10Extended", scenario="2023tilted"),
    Sample("RelValSingleMuPt100", scenario="2023GReco"),
    Sample("RelValSingleMuPt100", scenario="2023tilted"),
]

fastsimstartupsamples = [
    Sample('RelValTTbar', midfix="13", fastsim=True),
    Sample('RelValQCD_FlatPt_15_3000', midfix="13", fastsim=True),
    Sample('RelValSingleMuPt10', midfix="UP15", fastsim=True),
    Sample('RelValSingleMuPt100', midfix="UP15", fastsim=True)
]

pileupfastsimstartupsamples = [
    Sample('RelValTTbar', putype=putype("25ns"), punum=35, midfix="13", fastsim=True)
]

doFastVsFull = True
doPhase2PU = False
if "_pmx" in NewRelease:
    startupsamples = []
    fastsimstartupsamples = []
    doFastVsFull = False
    if not NewRelease in validation._globalTags:
        validation._globalTags[NewRelease] = validation._globalTags[NewRelease.replace("_pmx", "")]
if RefRelease is not None and "_pmx" in RefRelease:
    if not RefRelease in validation._globalTags:
        validation._globalTags[RefRelease] = validation._globalTags[RefRelease.replace("_pmx", "")]
if "_extended" in NewRelease:
    startupsamples = [
        Sample('RelValTTbar', midfix="13_HS"),
        Sample('RelValZMM', midfix="13_HS"),
    ]
    pileupstartupsamples = [
        Sample('RelValTTbar', putype=putype("25ns"), midfix="13_HS"),
        Sample('RelValZMM', putype=putype("25ns"), midfix="13_HS"),
    ]
    fastsimstartupsamples = []
    pileupfastsimstartupsamples = []
    doFastVsFull = False
    if not NewRelease in validation._globalTags:
        validation._globalTags[NewRelease] = validation._globalTags[NewRelease.replace("_extended", "")]
if "_phase1" in NewRelease:
    startupsamples = phase1samples
    pileupstartupsamples = []
    fastsimstartupsamples = []
    pileupfastsimstartupsamples = []
    doFastVsFull = False
if "_phase2" in NewRelease:
    startupsamples = phase2samples
    pileupstartupsamples = []
    fastsimstartupsamples = []
    pileupfastsimstartupsamples = []
    doFastVsFull = False
    doPhase2PU = True

### Track algorithm name and quality. Can be a list.
Algos= ['ootb', 'initialStep', 'lowPtTripletStep','pixelPairStep','detachedTripletStep','mixedTripletStep','pixelLessStep','tobTecStep','jetCoreRegionalStep','muonSeededStepInOut','muonSeededStepOutIn',
        'ak4PFJets','btvLike'
]
#Algos= ['ootb']
#Algos= ['ootb','initialStep','lowPtTripletStep','pixelPairStep','mixedTripletStep','muonSeededStepInOut','muonSeededStepOutIn'] # phase1
Qualities=['', 'highPurity']
VertexCollections=["offlinePrimaryVertices", "selectedOfflinePrimaryVertices"]

def limitProcessing(algo, quality):
    return algo in Algos and quality in Qualities

def limitRelVal(algo, quality): # for phase2 ATM
    return quality in ["", "highPurity"]

def ignore(a, q):
    return False

# Temporary until we have limited the set of histograms for phase2
kwargs_tracking = {}
if "_phase2" in NewRelease:
    kwargs_tracking["limitSubFoldersOnlyTo"] = {
        "": limitRelVal,
        "allTPEffic": ignore, "fromPV": ignore, "fromPVAllTP": ignore, # ignore for now to save disk space
        "seeding": ignore, "building": ignore # temporary until we have limited the set of histograms for phase2
    }



### Reference and new repository
RefRepository = '/afs/cern.ch/cms/Physics/tracking/validation/MC'
NewRepository = 'new' # copy output into a local folder

# Tracking validation plots
val = Validation(
    fullsimSamples = startupsamples + pileupstartupsamples,
    fastsimSamples = fastsimstartupsamples + pileupfastsimstartupsamples,
    refRelease=RefRelease, refRepository=RefRepository,
    newRelease=NewRelease, newRepository=NewRepository
)
htmlReport = val.createHtmlReport()
val.download()
val.doPlots(plotter=trackingPlots.plotter,
#            limitSubFoldersOnlyTo={"": limitProcessing, "allTPEffic": limitProcessing, "fromPV": limitProcessing, "fromPVAllTP": limitProcessing},
            htmlReport=htmlReport, doFastVsFull=doFastVsFull, doPhase2PU=doPhase2PU,
            **kwargs_tracking
)

val.download()
val.doPlots(plotter=vertexPlots.plotter,
            limitSubFoldersOnlyTo={"": VertexCollections},
            htmlReport=htmlReport, doFastVsFull=doFastVsFull, doPhase2PU=doPhase2PU,
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

