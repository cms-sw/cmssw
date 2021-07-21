from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
import Validation.RecoTrack.MultiTrackValidator_cfi
from Validation.RecoTrack.trajectorySeedTracks_cfi import trajectorySeedTracks as _trajectorySeedTracks
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
import Validation.RecoTrack.cutsRecoTracks_cfi as cutsRecoTracks_cfi
#from . import cutsRecoTracks_cfi

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.VertexAssociation.VertexAssociatorByPositionAndTracks_cfi import *
from CommonTools.RecoAlgos.trackingParticleRefSelector_cfi import trackingParticleRefSelector as _trackingParticleRefSelector
from CommonTools.RecoAlgos.trackingParticleConversionRefSelector_cfi import trackingParticleConversionRefSelector as _trackingParticleConversionRefSelector
from SimTracker.TrackHistory.trackingParticleBHadronRefSelector_cfi import trackingParticleBHadronRefSelector as _trackingParticleBHadronRefSelector
from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff import *
from CommonTools.RecoAlgos.recoChargedRefCandidateToTrackRefProducer_cfi import recoChargedRefCandidateToTrackRefProducer as _recoChargedRefCandidateToTrackRefProducer

import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
import RecoTracker.IterativeTracking.iterativeTkUtils as _utils
from Configuration.Eras.Modifier_fastSim_cff import fastSim
import six

### First define the stuff for the standard validation sequence
## Track selectors
for _eraName, _postfix, _era in _cfg.allEras():
    _seedProd = ["initialStepSeedsPreSplitting"]
    _trackProd = ["initialStepTracksPreSplitting"]
    if _eraName in ["trackingLowPU", "trackingPhase2PU140"]: # these don't have preSplitting
        _seedProd = []
        _trackProd = []

    locals()["_algos"+_postfix] = ["generalTracks"] + _cfg.iterationAlgos(_postfix) + ["duplicateMerge"]
    locals()["_seedProducersPreSplitting"+_postfix] = _seedProd
    locals()["_trackProducersPreSplitting"+_postfix] = _trackProd
    locals()["_seedProducers"+_postfix] = _cfg.seedProducers(_postfix)
    locals()["_trackProducers"+_postfix] = _cfg.trackProducers(_postfix)

    if _eraName != "trackingPhase2PU140":
        locals()["_electronSeedProducers"+_postfix] = ["tripletElectronSeeds", "pixelPairElectronSeeds", "stripPairElectronSeeds"]
    else:
        locals()["_electronSeedProducers"+_postfix] = ["tripletElectronSeeds"]

_removeForFastSimSeedProducers =["initialStepSeedsPreSplitting",
                                 "jetCoreRegionalStepSeeds",
                                 "jetCoreRegionalStepSeedsBarrel","jetCoreRegionalStepSeedsEndcap",
                                 "muonSeededSeedsInOut",
                                 "muonSeededSeedsOutIn"]

_seedProducers_fastSim = [ x for x in _seedProducers if x not in _removeForFastSimSeedProducers]

_removeForFastTrackProducers = ["initialStepTracksPreSplitting",
                                "jetCoreRegionalStepTracks",
                                "muonSeededTracksInOut",
                                "muonSeededTracksOutIn"]
_trackProducers_fastSim = [ x for x in _trackProducers if x not in _removeForFastTrackProducers]

def _algoToSelector(algo):
    sel = ""
    if algo != "generalTracks":
        sel = algo[0].upper()+algo[1:]
    return "cutsRecoTracks"+sel

def _addSelectorsByAlgo(algos, modDict):
    names = []
    task = cms.Task()
    for algo in algos:
        if algo == "generalTracks":
            continue
        modName = _algoToSelector(algo)
        if modName not in modDict:
            mod = cutsRecoTracks_cfi.cutsRecoTracks.clone(
#                src = [src],
                algorithm=[algo]
            )
            modDict[modName] = mod
        else:
            mod = modDict[modName]
        names.append(modName)
        task.add(mod)
    return (names, task)
def _addSelectorsByHp(algos, modDict):
    task = cms.Task()
    names = []
    for algo in algos:
        modName = _algoToSelector(algo)
        modNameHp = modName+"Hp"
        if modNameHp not in modDict:
            if algo == "generalTracks":
                mod = cutsRecoTracks_cfi.cutsRecoTracks.clone(quality=["highPurity"])
            else:
                mod = modDict[modName].clone(quality=["highPurity"])
            modDict[modNameHp] = mod
        else:
            mod = modDict[modNameHp]
        names.append(modNameHp)
        task.add(mod)
    return (names, task)
def _addSelectorsBySrc(modules, midfix, src, modDict):
    task = cms.Task()
    names = []
    for modName in modules:
        modNameNew = modName.replace("cutsRecoTracks", "cutsRecoTracks"+midfix)
        if modNameNew not in modDict:
            mod = modDict[modName].clone(src=src)
            modDict[modNameNew] = mod
        else:
            mod = modDict[modNameNew]
        names.append(modNameNew)
        task.add(mod)
    return (names, task)
def _addSelectorsByOriginalAlgoMask(modules, midfix, algoParam,modDict):
    task = cms.Task()
    names = []
    for modName in modules:
        if modName[-2:] == "Hp":
            modNameNew = modName[:-2] + midfix + "Hp"
        else:
            modNameNew = modName + midfix
        if modNameNew not in modDict:
            mod = modDict[modName].clone()
            setattr(mod, algoParam, mod.algorithm.value())
            mod.algorithm = []
            modDict[modNameNew] = mod
        else:
            mod = modDict[modNameNew]
        names.append(modNameNew)
        task.add(mod)
    return (names, task)
def _addSeedToTrackProducers(seedProducers,modDict):
    names = []
    task = cms.Task()
    for seed in seedProducers:
        modName = "seedTracks"+seed
        if modName not in modDict:
            mod = _trajectorySeedTracks.clone(src=seed)
            modDict[modName] = mod
        else:
            mod = modDict[modName]
        names.append(modName)
        task.add(mod)
    return (names, task)

_relevantEras = _cfg.allEras()
_relevantErasAndFastSim = _relevantEras + [("fastSim", "_fastSim", fastSim)]
def _translateArgs(args, postfix, modDict):
    ret = []
    for arg in args:
        if isinstance(arg, list):
            ret.append(_translateArgs(arg, postfix, modDict))
        else:
            ret.append(modDict[arg+postfix])
    return ret
def _taskForEachEra(function, args, names, task, modDict, plainArgs=[], modifyTask=None, includeFastSim=False):
    if task[0] != "_":
        raise Exception("Task name is expected to begin with _")

    _eras = _relevantErasAndFastSim if includeFastSim else _relevantEras
    for eraName, postfix, _era in _eras:
        _args = _translateArgs(args, postfix, modDict)
        _args.extend(plainArgs)
        ret = function(*_args, modDict=modDict)
        if len(ret) != 2:
            raise Exception("_taskForEachEra is expected to return 2 values, but function returned %d" % len(ret))
        modDict[names+postfix] = ret[0]
        modDict[task+postfix] = ret[1]

    # The task of the first era will be the default one
    defaultTaskName = task+_eras[0][0]
    defaultTask = modDict[defaultTaskName]
    modDict[defaultTaskName[1:]] = defaultTask # remove leading underscore

    # Optionally modify task before applying the era
    if modifyTask is not None:
        for eraName, postfix, _era in _eras:
            modifyTask(modDict[task+postfix])

    # Apply eras
    for _eraName, _postfix, _era in _eras[1:]:
        _era.toReplaceWith(defaultTask, modDict[task+_postfix])
def _setForEra(module, eraName, era, **kwargs):
    if eraName == "":
        for key, value in six.iteritems(kwargs):
            setattr(module, key, value)
    else:
        era.toModify(module, **kwargs)

# Seeding layer sets
def _getSeedingLayers(seedProducers, config):
    def _findSeedingLayers(name):
        prod = getattr(config, name)
        if hasattr(prod, "triplets"):
            if hasattr(prod, "layerList"): # merger
                return prod.layerList.refToPSet_.value()
            return _findSeedingLayers(prod.triplets.getModuleLabel())
        elif hasattr(prod, "doublets"):
            return _findSeedingLayers(prod.doublets.getModuleLabel())
        label = prod.trackingRegionsSeedingLayers.getModuleLabel()
        if label != "":
            return label
        return prod.seedingLayers.getModuleLabel()

    seedingLayersMerged = []
    for seedName in seedProducers:
        seedProd = getattr(config, seedName)
        seedingLayersName = None
        seedingLayers = None
        if hasattr(seedProd, "OrderedHitsFactoryPSet"): # old seeding framework
            seedingLayersName = seedProd.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()
        elif hasattr(seedProd, "seedingHitSets"): # new seeding framework
            seedingLayersName = _findSeedingLayers(seedProd.seedingHitSets.getModuleLabel())
        elif hasattr(seedProd, "layerList"): # FastSim:
            seedingLayers = seedProd.layerList.value()
        else:
            continue

        if seedingLayersName is not None:
            seedingLayers = getattr(config, seedingLayersName).layerList.value()
        for layerSet in seedingLayers:
            if layerSet not in seedingLayersMerged:
                seedingLayersMerged.append(layerSet)

    return seedingLayersMerged
import RecoTracker.IterativeTracking.iterativeTk_cff as _iterativeTk_cff
import RecoTracker.IterativeTracking.ElectronSeeds_cff as _ElectronSeeds_cff
for _eraName, _postfix, _era in _relevantErasAndFastSim:
    _stdLayers = _getSeedingLayers(locals()["_seedProducers"+_postfix], _iterativeTk_cff)
    _eleLayers = []
    if "_electronSeedProducers"+_postfix in locals(): # doesn't exist for FastSim
        for _layer in _getSeedingLayers(locals()["_electronSeedProducers"+_postfix], _ElectronSeeds_cff):
            if _layer not in _stdLayers:
                _eleLayers.append(_layer)

    locals()["_seedingLayerSets"+_postfix] = _stdLayers
    locals()["_seedingLayerSetsForElectrons"+_postfix] = _eleLayers


# MVA selectors
def _getMVASelectors(postfix):
    mvaSel = _utils.getMVASelectors(postfix)

    pset = cms.untracked.PSet()
    for iteration, (trackProducer, classifiers) in six.iteritems(mvaSel):
        setattr(pset, trackProducer, cms.untracked.vstring(classifiers))
    return pset
for _eraName, _postfix, _era in _relevantEras:
    locals()["_mvaSelectors"+_postfix] = _getMVASelectors(_postfix)

# Validation iterative steps
_taskForEachEra(_addSelectorsByAlgo, args=["_algos"], names="_selectorsByAlgo", task="_tracksValidationSelectorsByAlgo", modDict=globals())

# high purity
_taskForEachEra(_addSelectorsByHp, args=["_algos"], names="_selectorsByAlgoHp", task="_tracksValidationSelectorsByAlgoHp", modDict=globals())

# by originalAlgo
for _eraName, _postfix, _era in _relevantEras:
    locals()["_selectorsByAlgoAndHp"+_postfix] = locals()["_selectorsByAlgo"+_postfix] + locals()["_selectorsByAlgoHp"+_postfix]
    # For ByAlgoMask
    locals()["_selectorsByAlgoAndHpNoGenTk"+_postfix] = [n for n in locals()["_selectorsByAlgoAndHp"+_postfix] if n not in ["generalTracks", "cutsRecoTracksHp"]]
    # For ByOriginalAlgo
    locals()["_selectorsByAlgoAndHpNoGenTkDupMerge"+_postfix] = [n for n in locals()["_selectorsByAlgoAndHpNoGenTk"+_postfix] if n not in ["cutsRecoTracksDuplicateMerge", "cutsRecoTracksDuplicateMergeHp"]]
_taskForEachEra(_addSelectorsByOriginalAlgoMask, modDict = globals(),
                    args = ["_selectorsByAlgoAndHpNoGenTkDupMerge"], plainArgs = ["ByOriginalAlgo", "originalAlgorithm"],
                    names = "_selectorsByOriginalAlgo", task = "_tracksValidationSelectorsByOriginalAlgo")


for _eraName, _postfix, _era in _relevantEras:
    selectors = locals()["_selectorsByAlgoHp"+_postfix]
    locals()["_generalTracksHp"+_postfix] = selectors[0]
    locals()["_selectorsByAlgoHp"+_postfix] = selectors[1:]

# BTV-like selection
import PhysicsTools.RecoAlgos.btvTracks_cfi as btvTracks_cfi
cutsRecoTracksBtvLike = btvTracks_cfi.btvTrackRefs.clone()

# Select tracks associated to AK4 jets
import RecoJets.JetAssociationProducers.ak4JTA_cff as ak4JTA_cff
ak4JetTracksAssociatorExplicitAll = ak4JTA_cff.ak4JetTracksAssociatorExplicit.clone(
    jets = "ak4PFJets"
)
from JetMETCorrections.Configuration.JetCorrectors_cff import *
import JetMETCorrections.JetCorrector.jetTracksAssociationToTrackRefs_cfi as jetTracksAssociationToTrackRefs_cfi
cutsRecoTracksAK4PFJets = jetTracksAssociationToTrackRefs_cfi.jetTracksAssociationToTrackRefs.clone(
    association = "ak4JetTracksAssociatorExplicitAll",
    jets = "ak4PFJets",
    correctedPtMin = 10,
)


## Select signal TrackingParticles, and do the corresponding associations
trackingParticlesSignal = _trackingParticleRefSelector.clone(
    signalOnly = True,
    chargedOnly = False,
    tip = 1e5,
    lip = 1e5,
    minRapidity = -10,
    maxRapidity = 10,
    ptMin = 0,
)
## Select in-time TrackingParticles, and do the corresponding associations
trackingParticlesInTime = trackingParticlesSignal.clone(
    signalOnly = False,
    intimeOnly = True,
)

# select tracks with pT > 0.9 GeV (for upgrade fake rates)
generalTracksPt09 = cutsRecoTracks_cfi.cutsRecoTracks.clone(ptMin=0.9)
# and then the selectors
_taskForEachEra(_addSelectorsBySrc, modDict=globals(),
                args=[["_generalTracksHp"]],
                plainArgs=["Pt09", "generalTracksPt09"],
                names="_selectorsPt09", task="_tracksValidationSelectorsPt09",
                modifyTask=lambda task:task.add(generalTracksPt09))

# select tracks from the PV
from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import trackWithVertexRefSelector as _trackWithVertexRefSelector
generalTracksFromPV = _trackWithVertexRefSelector.clone(
    src = "generalTracks",
    ptMin = 0,
    ptMax = 1e10,
    ptErrorCut = 1e10,
    quality = "loose",
    vertexTag = "offlinePrimaryVertices",
    nVertices = 1,
    vtxFallback = False,
    zetaVtx = 0.1, # 1 mm
    rhoVtx = 1e10, # intentionally no dxy cut
)
# and then the selectors
_taskForEachEra(_addSelectorsBySrc, modDict=globals(),
                    args=[["_generalTracksHp"]],
                    plainArgs=["FromPV", "generalTracksFromPV"],
                    names="_selectorsFromPV", task="_tracksValidationSelectorsFromPV",
                    modifyTask=lambda task: task.add(generalTracksFromPV))

# select tracks with pT > 0.9 GeV from the PV
generalTracksFromPVPt09 = generalTracksPt09.clone(src="generalTracksFromPV")
# and then the selectors
_taskForEachEra(_addSelectorsBySrc, modDict=globals(),
                args=[["_generalTracksHp"]],
                plainArgs=["FromPVPt09", "generalTracksFromPVPt09"],
                names="_selectorsFromPVPt09", task="_tracksValidationSelectorsFromPVPt09",
                modifyTask=lambda task: task.add(generalTracksFromPVPt09))

## Select conversion TrackingParticles, and define the corresponding associator
trackingParticlesConversion = _trackingParticleConversionRefSelector.clone()

## Select electron TPs
trackingParticlesElectron = _trackingParticleRefSelector.clone(
    pdgId = [-11, 11],
    signalOnly = False,
    tip = 1e5,
    lip = 1e5,
    minRapidity = -10,
    maxRapidity = 10,
    ptMin = 0,
)

# Select jets for JetCore tracking
highPtJets = cms.EDFilter("CandPtrSelector", src = cms.InputTag("ak4CaloJets"), cut = cms.string("pt()>1000")) 
highPtJetsForTrk = highPtJets.clone(src = "ak4CaloJetsForTrk")

# Select B-hadron TPs
trackingParticlesBHadron = _trackingParticleBHadronRefSelector.clone()

## MTV instances
trackValidator = Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone(
    useLogPt = cms.untracked.bool(True),
    dodEdxPlots = True,
    doPVAssociationPlots = True
    #,minpT = cms.double(-1)
    #,maxpT = cms.double(3)
    #,nintpT = cms.int32(40)
)
fastSim.toModify(trackValidator,
                      dodEdxPlots = False)

for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidator, _eraName, _era,
               label = ["generalTracks", locals()["_generalTracksHp"+_postfix]] +
                       locals()["_selectorsByAlgo"+_postfix] + locals()["_selectorsByAlgoHp"+_postfix] +
                       locals()["_selectorsByOriginalAlgo"+_postfix] +
                       ["generalTracksPt09"] + locals()["_selectorsPt09"+_postfix] +
               [
                   "cutsRecoTracksBtvLike",
                   "cutsRecoTracksAK4PFJets"
               ],
               doResolutionPlotsForLabels = [
                   "generalTracks",
                   locals()["_generalTracksHp"+_postfix],
                   "generalTracksPt09",
                   "cutsRecoTracksBtvLike",
                   "cutsRecoTracksJetCoreRegionalStepByOriginalAlgo",
               ]
    )
    _setForEra(trackValidator.histoProducerAlgoBlock, _eraName, _era, seedingLayerSets=locals()["_seedingLayerSets"+_postfix])

# for low-pT
trackValidatorTPPtLess09 = trackValidator.clone(
    dirName = "Tracking/TrackTPPtLess09/",
    label = [x for x in trackValidator.label.value() if ("Pt09" not in x) and ("BtvLike" not in x) and ("AK4PFJets" not in x)],
    ptMaxTP = 0.9, # set maximum pT globally
    histoProducerAlgoBlock = dict(
        TpSelectorForEfficiencyVsEta  = dict(ptMin=0.05), # enough to set min pT here
        TpSelectorForEfficiencyVsPhi  = dict(ptMin=0.05),
        TpSelectorForEfficiencyVsVTXR = dict(ptMin=0.05),
        TpSelectorForEfficiencyVsVTXZ = dict(ptMin=0.05),
    ),
    doSimPlots = False,       # same as in trackValidator, no need to repeat here
    doRecoTrackPlots = False, # fake rates are same as in trackValidator, no need to repeat here
    doResolutionPlotsForLabels = ["disabled"], # resolutions are same as in trackValidator, no need to repeat here
)

## Select signal TrackingParticles, and do the corresponding associations
trackingParticlesEtaGreater2p7 = _trackingParticleRefSelector.clone(
    signalOnly = cms.bool(False),
    tip = 1e5,
    lip = 1e5,
    minRapidity = -2.7,
    maxRapidity =  2.7,
    invertRapidityCut = cms.bool(True),
    ptMin = 0,
)


# select tracks with |eta| > 2.7
generalTracksEtaGreater2p7 = cutsRecoTracks_cfi.cutsRecoTracks.clone(
    minRapidity = cms.double(-2.7),
    maxRapidity = cms.double( 2.7),
    invertRapidityCut = cms.bool(True)
)

_taskForEachEra(_addSelectorsBySrc, modDict=globals(),
                    args=[["_generalTracksHp"]],
                    plainArgs=["EtaGreater2p7", "generalTracksEtaGreater2p7"],
                    names="_selectorsEtaGreater2p7", task="_tracksValidationSelectorsEtaGreater2p7",
                    modifyTask=lambda task: task.add(generalTracksEtaGreater2p7))

# for high-eta (phase2 : |eta| > 2.7)
trackValidatorTPEtaGreater2p7 = trackValidator.clone(
    dirName = "Tracking/TrackTPEtaGreater2p7/",
    label_tp_effic = "trackingParticlesEtaGreater2p7",
    label_tp_fake  = "trackingParticlesEtaGreater2p7",
    label_tp_effic_refvector = True,
    label_tp_fake_refvector  = True,
    dodEdxPlots = False,
#    doPVAssociationPlots = False,
    minRapidityTP = -2.7,
    maxRapidityTP = 2.7,
    invertRapidityCutTP = True,
#    ptMaxTP = 0.9, # set maximum pT globally
    histoProducerAlgoBlock = dict(
        TpSelectorForEfficiencyVsPt   = dict(ptMin=0.005,minRapidity=-2.7,maxRapidity=2.7,invertRapidityCut=True), # enough to set min pT here
        TpSelectorForEfficiencyVsEta  = dict(ptMin=0.005,minRapidity=-2.7,maxRapidity=2.7,invertRapidityCut=True), # enough to set min pT here
        TpSelectorForEfficiencyVsPhi  = dict(ptMin=0.005,minRapidity=-2.7,maxRapidity=2.7,invertRapidityCut=True),
        TpSelectorForEfficiencyVsVTXR = dict(ptMin=0.005,minRapidity=-2.7,maxRapidity=2.7,invertRapidityCut=True),
        TpSelectorForEfficiencyVsVTXZ = dict(ptMin=0.005,minRapidity=-2.7,maxRapidity=2.7,invertRapidityCut=True),
        generalTpSelector             = dict(ptMin=0.005,minRapidity=-2.7,maxRapidity=2.7,invertRapidityCut=True),
#        minEta  = -4.5,
#        maxEta  =  4.5,
#        nintEta = 90,
        #    minPt  = 0.01,
    ),
    doSimPlots = True,       # ####same as in trackValidator, no need to repeat here
    doRecoTrackPlots = True, # ####fake rates are same as in trackValidator, no need to repeat here
    doResolutionPlotsForLabels = ["disabled"] # resolutions are same as in trackValidator, no need to repeat here
)
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorTPEtaGreater2p7, _eraName, _era,
               label = ["generalTracksEtaGreater2p7"] + locals()["_selectorsEtaGreater2p7"+_postfix] +
                       locals()["_selectorsByAlgo"+_postfix] + locals()["_selectorsByAlgoHp"+_postfix],
               doResolutionPlotsForLabels = ["generalTracksEtaGreater2p7"] + locals()["_selectorsEtaGreater2p7"+_postfix]
    )

# For efficiency of signal TPs vs. signal tracks, and fake rate of
# signal tracks vs. signal TPs
trackValidatorFromPV = trackValidator.clone(
    dirName = "Tracking/TrackFromPV/",
    label_tp_effic = "trackingParticlesSignal",
    label_tp_fake = "trackingParticlesSignal",
    label_tp_effic_refvector = True,
    label_tp_fake_refvector = True,
    trackCollectionForDrCalculation = "generalTracksFromPV",
    doPlotsOnlyForTruePV = True,
    doPVAssociationPlots = False,
    doResolutionPlotsForLabels = ["disabled"],
)
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorFromPV, _eraName, _era,
               label = ["generalTracksFromPV"] + locals()["_selectorsFromPV"+_postfix] + ["generalTracksFromPVPt09"] + locals()["_selectorsFromPVPt09"+_postfix],
               doResolutionPlotsForLabels = [] # for standard "FromPV" do resolution plots for all input collections as they are already limited
    )

# For fake rate of signal tracks vs. all TPs, and pileup rate of
# signal tracks vs. non-signal TPs
trackValidatorFromPVAllTP = trackValidatorFromPV.clone(
    dirName = "Tracking/TrackFromPVAllTP/",
    label_tp_effic = trackValidator.label_tp_effic.value(),
    label_tp_fake = trackValidator.label_tp_fake.value(),
    label_tp_effic_refvector = False,
    label_tp_fake_refvector = False,
    doSimPlots = False,
    doSimTrackPlots = False,
    doResolutionPlotsForLabels = ["disabled"], # resolution plots are the same as in "trackValidatorFromPV"
)

# For efficiency of all TPs vs. all tracks
trackValidatorAllTPEffic = trackValidator.clone(
    dirName = "Tracking/TrackAllTPEffic/",
    label = [x for x in trackValidator.label.value() if "Pt09" not in x],
    doSimPlots = False,
    doRecoTrackPlots = True, # Fake rate of all tracks vs. all TPs is already included in trackValidator, but we want the reco plots for other reasons
    doPVAssociationPlots = False,
    doResolutionPlotsForLabels = ["disabled"], # resolution plots are the same as in "trackValidator"
)
trackValidatorAllTPEffic.histoProducerAlgoBlock.generalTpSelector.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsEta.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPhi.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXR.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXZ.signalOnly = False
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorAllTPEffic, _eraName, _era, label = ["generalTracks", locals()["_generalTracksHp"+_postfix]])

# Built tracks, in the standard sequence mainly for monitoring the track selection MVA
tpClusterProducerPreSplitting = tpClusterProducer.clone(pixelClusterSrc = "siPixelClustersPreSplitting")
quickTrackAssociatorByHitsPreSplitting = quickTrackAssociatorByHits.clone(cluster2TPSrc = "tpClusterProducerPreSplitting")
_trackValidatorSeedingBuilding = trackValidator.clone( # common for built tracks and seeds (in trackingOnly)
    associators = ["quickTrackAssociatorByHits"],
    UseAssociators = True,
    dodEdxPlots = False,
    doPVAssociationPlots = False,
    doSimPlots = False,
    doResolutionPlotsForLabels = ["disabled"],
)
trackValidatorBuilding = _trackValidatorSeedingBuilding.clone(
    dirName = "Tracking/TrackBuilding/",
    doMVAPlots = True,
    doResolutionPlotsForLabels = ['jetCoreRegionalStepTracks'],
)
trackValidatorBuildingPreSplitting = trackValidatorBuilding.clone(
    associators = ["quickTrackAssociatorByHitsPreSplitting"],
    doMVAPlots = False,
    doSummaryPlots = False,
)
for _eraName, _postfix, _era in _relevantErasAndFastSim:
    _setForEra(trackValidatorBuilding, _eraName, _era, label = locals()["_trackProducers"+_postfix])
fastSim.toModify(trackValidatorBuilding, doMVAPlots=False)
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorBuilding, _eraName, _era, mvaLabels = locals()["_mvaSelectors"+_postfix])
    _setForEra(trackValidatorBuildingPreSplitting, _eraName, _era, label = locals()["_trackProducersPreSplitting"+_postfix])


# For conversions
trackValidatorConversion = trackValidator.clone(
    dirName = "Tracking/TrackConversion/",
    label = [
        "convStepTracks",
        "conversionStepTracks",
        "ckfInOutTracksFromConversions",
        "ckfOutInTracksFromConversions",
    ],
    label_tp_effic = "trackingParticlesConversion",
    label_tp_effic_refvector = True,
    associators = ["quickTrackAssociatorByHits"],
    UseAssociators = True,
    doSimPlots = True,
    dodEdxPlots = False,
    doPVAssociationPlots = False,
    calculateDrSingleCollection = False,
)
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import convLayerPairs as _convLayerPairs
def _uniqueFirstLayers(layerList):
    firstLayers = [layerSet.split("+")[0] for layerSet in layerList]
    ret = []
    for l in firstLayers:
        if not l in ret:
            ret.append(l)
            # For conversions add also the mono-TEC to the list as 'TEC'
            # is used for both matched and unmatched rphi/stereo hits
            if l.startswith("TEC"):
                ret.append("M"+l)
    return ret
# PhotonConversionTrajectorySeedProducerFromSingleLeg keeps only the
# first hit of the pairs in the seed, bookkeeping those is the best we
# can do without major further development
trackValidatorConversion.histoProducerAlgoBlock.seedingLayerSets = _uniqueFirstLayers(_convLayerPairs.layerList.value())
# relax lip and tip
for n in ["Eta", "Phi", "Pt", "VTXR", "VTXZ"]:
    pset = getattr(trackValidatorConversion.histoProducerAlgoBlock, "TpSelectorForEfficiencyVs"+n)
    pset.lip = trackValidatorConversion.lipTP.value()
    pset.tip = trackValidatorConversion.tipTP.value()

# For electrons
trackValidatorGsfTracks = trackValidatorConversion.clone(
    dirName = "Tracking/TrackGsf/",
    label = ["electronGsfTracks"],
    label_tp_effic = "trackingParticlesElectron",
)
# add the additional seeding layers from ElectronSeeds
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorGsfTracks.histoProducerAlgoBlock, _eraName, _era, seedingLayerSets=trackValidator.histoProducerAlgoBlock.seedingLayerSets.value()+locals()["_seedingLayerSetsForElectrons"+_postfix])

# For jetCore tracks
trackValidatorJetCore = trackValidator.clone(#equivalent to trackBuilding case
    dirName = "Tracking/JetCore/",
    useLogPt = cms.untracked.bool(True),
    dodEdxPlots = False,
    associators= ["trackAssociatorByChi2"],#cms.untracked.VInputTag('MTVTrackAssociationByChi2'),
    UseAssociators = True,
    doPVAssociationPlots = True,
    label_tp_effic = "trackingParticlesInTime",
    label_tp_fake = "trackingParticlesInTime",
    label_tp_effic_refvector = True,
    label_tp_fake_refvector = True,
)
for _eraName, _postfix, _era in _relevantEras:
    if 'jetCoreRegionalStep' in _cfg.iterationAlgos(_postfix) :
        _setForEra(trackValidatorJetCore, _eraName, _era,
            label = ["generalTracks", "jetCoreRegionalStepTracks", 
                    "cutsRecoTracksJetCoreRegionalStepByOriginalAlgo","cutsRecoTracksJetCoreRegionalStepByOriginalAlgoHp",
                    "cutsRecoTracksJetCoreRegionalStep", "cutsRecoTracksJetCoreRegionalStepHp"],
            doResolutionPlotsForLabels =["generalTracks", "jetCoreRegionalStepTracks", 
                    "cutsRecoTracksJetCoreRegionalStepByOriginalAlgo","cutsRecoTracksJetCoreRegionalStepByOriginalAlgoHp",
                    "cutsRecoTracksJetCoreRegionalStep", "cutsRecoTracksJetCoreRegionalStepHp"], 
        )

# for B-hadrons
trackValidatorBHadron = trackValidator.clone(
    dirName = "Tracking/TrackBHadron/",
    label_tp_effic = "trackingParticlesBHadron",
    label_tp_effic_refvector = True,
    doSimPlots = True,
    doRecoTrackPlots = False, # Fake rate is defined wrt. all TPs, and that is already included in trackValidator
    dodEdxPlots = False,
)
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorBHadron, _eraName, _era,
               label = ["generalTracks", locals()["_generalTracksHp"+_postfix], "cutsRecoTracksBtvLike"]
    )


# for displaced tracks
trackValidatorDisplaced = trackValidator.clone(
    dirName = "Tracking/TrackDisplaced/",
    label = [x for x in trackValidator.label.value() if ("Pt09" not in x) and ("BtvLike" not in x) and ("AK4PFJets" not in x)],
    ptMaxTP = 1e5,
    dodEdxPlots = False,
    invertRapidityCutTP = False,
    histoProducerAlgoBlock = dict(
        TpSelectorForEfficiencyVsPt   = dict(ptMin=0.005, signalOnly=True, tip=1e5, lip=1e5), # enough to set min pT here
        TpSelectorForEfficiencyVsEta  = dict(ptMin=0.005, signalOnly=True, tip=1e5, lip=1e5), # enough to set min pT here
        TpSelectorForEfficiencyVsPhi  = dict(ptMin=0.005, signalOnly=True, tip=1e5, lip=1e5),
        TpSelectorForEfficiencyVsVTXR = dict(ptMin=0.005, signalOnly=True, tip=1e5, lip=1e5),
        TpSelectorForEfficiencyVsVTXZ = dict(ptMin=0.005, signalOnly=True, tip=1e5, lip=1e5),
        generalTpSelector             = dict(ptMin=0.005, signalOnly=True, tip=1e5, lip=1e5),
        minDxy = -60,
        maxDxy =  60,
        nintDxy = 120,
        minDz  = -30,
        maxDz  =  30,
        nintDz =  60,
    ),
    signalOnlyTP = True,
    lipTP = 1e5,
    tipTP = 1e5,
)

# the track selectors
tracksValidationSelectors = cms.Task(
    tracksValidationSelectorsByAlgo,
    tracksValidationSelectorsByAlgoHp,
    tracksValidationSelectorsByOriginalAlgo,
    cutsRecoTracksBtvLike,
    ak4JetTracksAssociatorExplicitAll,
    cutsRecoTracksAK4PFJets
)
phase2_tracker.toModify(tracksValidationSelectors, lambda x: x.add(generalTracksEtaGreater2p7))
phase2_tracker.toModify(tracksValidationSelectors, lambda x: x.add(cutsRecoTracksEtaGreater2p7Hp))

# Validation iterative steps
_taskForEachEra(_addSelectorsByAlgo, modDict=globals(),
                args=["_algos"], 
                names="_selectorsByAlgo", task="_tracksEtaGreater2p7ValidationSelectorsByAlgo"                
               )

# high purity
_taskForEachEra(_addSelectorsByHp, modDict=globals(),
                args=["_algos"], 
                names="_selectorsByAlgoHp", task="_tracksEtaGreater2p7ValidationSelectorsByAlgoHp"
               )

for _eraName, _postfix, _era in _relevantEras:
    selectors = locals()["_selectorsByAlgoHp"+_postfix]
    locals()["_generalTracksHp"+_postfix] = selectors[0]
    locals()["_selectorsByAlgoHp"+_postfix] = selectors[1:]

phase2_tracker.toModify(tracksValidationSelectors, lambda x: x.add(tracksEtaGreater2p7ValidationSelectorsByAlgo))
phase2_tracker.toModify(tracksValidationSelectors, lambda x: x.add(tracksEtaGreater2p7ValidationSelectorsByAlgoHp))

tracksValidationTruth = cms.Task(
    tpClusterProducer,
    tpClusterProducerPreSplitting,
    trackAssociatorByChi2, 
    quickTrackAssociatorByHits,
    quickTrackAssociatorByHitsPreSplitting,
    trackingParticleRecoTrackAsssociation,
    VertexAssociatorByPositionAndTracks,
    trackingParticleNumberOfLayersProducer
)
fastSim.toModify(tracksValidationTruth, lambda x: x.remove(tpClusterProducer))

tracksPreValidation = cms.Task(
    highPtJetsForTrk,
    tracksValidationSelectors,
    tracksValidationSelectorsPt09,
    tracksValidationSelectorsFromPV,
    tracksValidationSelectorsFromPVPt09,
    tracksValidationTruth,
    trackingParticlesSignal,
    trackingParticlesInTime,
    trackingParticlesElectron,
    trackingParticlesConversion
)
fastSim.toReplaceWith(tracksPreValidation, tracksPreValidation.copyAndExclude([
    trackingParticlesElectron,
    trackingParticlesConversion,
]))



tracksValidation = cms.Sequence(
    trackValidator +
    trackValidatorTPPtLess09 +
    trackValidatorFromPV +
    trackValidatorFromPVAllTP +
    trackValidatorAllTPEffic +
    trackValidatorBuilding +
    trackValidatorBuildingPreSplitting +
    trackValidatorConversion +
    trackValidatorGsfTracks,
    tracksPreValidation
)

from Configuration.ProcessModifiers.seedingDeepCore_cff import seedingDeepCore
seedingDeepCore.toReplaceWith(tracksValidation, cms.Sequence(tracksValidation.copy()+trackValidatorJetCore))

from Configuration.ProcessModifiers.displacedTrackValidation_cff import displacedTrackValidation
displacedTrackValidation.toReplaceWith(tracksValidation, cms.Sequence(tracksValidation.copy()+trackValidatorDisplaced))

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
tracksPreValidationPhase2 = tracksPreValidation.copy()
tracksPreValidationPhase2.add(trackingParticlesEtaGreater2p7)
phase2_tracker.toReplaceWith(tracksPreValidation, tracksPreValidationPhase2)

tracksValidationPhase2 = tracksValidation.copyAndExclude([
    trackValidatorJetCore
])
tracksValidationPhase2+=trackValidatorTPEtaGreater2p7
phase2_tracker.toReplaceWith(tracksValidation, tracksValidationPhase2)

fastSim.toReplaceWith(tracksValidation, tracksValidation.copyAndExclude([
    trackValidatorBuildingPreSplitting,
    trackValidatorConversion,
    trackValidatorGsfTracks,
]))

### Then define stuff for standalone mode (i.e. MTV with RECO+DIGI input)

# Select by originalAlgo and algoMask
_taskForEachEra(_addSelectorsByOriginalAlgoMask, modDict = globals(),
                args = ["_selectorsByAlgoAndHpNoGenTk"], plainArgs = ["ByAlgoMask", "algorithmMaskContains"],
                names = "_selectorsByAlgoMask", task = "_tracksValidationSelectorsByAlgoMaskStandalone")

# Select pT>0.9 by iteration
# Need to avoid generalTracks+HP because those are already included in the standard validator
_taskForEachEra(_addSelectorsBySrc, modDict = globals(),
                args = ["_selectorsByAlgoAndHpNoGenTk"], plainArgs = ["Pt09", "generalTracksPt09"],
                names = "_selectorsPt09Standalone", task = "_tracksValidationSelectorsPt09Standalone")

# Select fromPV by iteration
# Need to avoid generalTracks+HP because those are already included in the standard validator
_taskForEachEra(_addSelectorsBySrc, modDict = globals(),
                args = ["_selectorsByAlgoAndHpNoGenTk"], plainArgs = ["FromPV", "generalTracksFromPV"],
                names = "_selectorsFromPVStandalone", task = "_tracksValidationSelectorsFromPVStandalone")

# Select pt>0.9 and fromPV by iteration
# Need to avoid generalTracks+HP because those are already included in the standard validator
_taskForEachEra(_addSelectorsBySrc, modDict = globals(),
                args = ["_selectorsByAlgoAndHpNoGenTk"], plainArgs = ["FromPVPt09", "generalTracksFromPVPt09"],
                names = "_selectorsFromPVPt09Standalone", task = "_tracksValidationSelectorsFromPVPt09Standalone")

# MTV instances
trackValidatorStandalone = trackValidator.clone(
    cores = "highPtJets"
)
trackValidatorTPPtLess09Standalone = trackValidatorTPPtLess09.clone(
    cores = "highPtJets"
)
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorStandalone, _eraName, _era, label = trackValidator.label + locals()["_selectorsByAlgoMask"+_postfix] + locals()["_selectorsPt09Standalone"+_postfix])
    _setForEra(trackValidatorTPPtLess09Standalone, _eraName, _era, label = trackValidatorTPPtLess09.label + locals()["_selectorsByAlgoMask"+_postfix] + locals()["_selectorsPt09Standalone"+_postfix])

trackValidatorFromPVStandalone = trackValidatorFromPV.clone(
    cores = "highPtJets"
)
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorFromPVStandalone, _eraName, _era, label = trackValidatorFromPV.label + locals()["_selectorsFromPVStandalone"+_postfix] + locals()["_selectorsFromPVPt09Standalone"+_postfix])
# do resolutions as in the standard version

trackValidatorFromPVAllTPStandalone = trackValidatorFromPVAllTP.clone(
    label = trackValidatorFromPVStandalone.label.value(),
    cores = "highPtJets"

)
trackValidatorAllTPEfficStandalone = trackValidatorAllTPEffic.clone(
    label = [ x for x in trackValidator.label.value() if x not in ["cutsRecoTracksBtvLike", "cutsRecoTracksAK4PFJets"] and "Pt09" not in x],
    cores = "highPtJets"
)

trackValidatorConversionStandalone = trackValidatorConversion.clone(
    label = [x for x in trackValidatorConversion.label if x != "convStepTracks"],
    cores = "highPtJets"
)

trackValidatorBHadronStandalone = trackValidatorBHadron.clone(
    label = [x for x in trackValidatorStandalone.label if "Pt09" not in x],
    cores = "highPtJets"
)

trackValidatorGsfTracksStandalone = trackValidatorGsfTracks.clone(
    cores = "highPtJets"
)

# sequences
tracksPreValidationStandalone = tracksPreValidation.copy()
tracksPreValidationStandalone.add(trackingParticlesBHadron)
tracksPreValidationStandalone.replace(highPtJetsForTrk,highPtJets)
fastSim.toReplaceWith(tracksPreValidationStandalone, tracksPreValidation)

tracksValidationSelectorsStandalone = cms.Task(
    tracksValidationSelectorsByAlgoMaskStandalone,
    tracksValidationSelectorsPt09Standalone,
    tracksValidationSelectorsFromPVStandalone,
    tracksValidationSelectorsFromPVPt09Standalone
)

# we copy this for both Standalone and TrackingOnly
#  and later make modifications from it which change based on era
_trackValidatorsBase = cms.Sequence(
    trackValidatorStandalone +
    trackValidatorTPPtLess09Standalone +
    trackValidatorFromPVStandalone +
    trackValidatorFromPVAllTPStandalone +
    trackValidatorAllTPEfficStandalone +
    trackValidatorConversionStandalone +
    trackValidatorGsfTracksStandalone +
    trackValidatorBHadronStandalone
)

_trackValidatorsBasePhase2 = _trackValidatorsBase.copy()
_trackValidatorsBasePhase2+=trackValidatorTPEtaGreater2p7
phase2_tracker.toReplaceWith(_trackValidatorsBase, _trackValidatorsBasePhase2)

trackValidatorsStandalone = _trackValidatorsBase.copy()
fastSim.toModify(trackValidatorsStandalone, lambda x: x.remove(trackValidatorConversionStandalone) )

tracksValidationStandalone = cms.Sequence(
    ak4PFL1FastL2L3CorrectorChain +
    trackValidatorsStandalone,
    tracksPreValidationStandalone,
    tracksValidationSelectorsStandalone
)

### TrackingOnly mode (i.e. MTV with DIGI input + tracking-only reconstruction)

# selectors
tracksValidationSelectorsTrackingOnly = tracksValidationSelectors.copyAndExclude([ak4JetTracksAssociatorExplicitAll,cutsRecoTracksAK4PFJets]) # selectors using track information only (i.e. no PF)
_taskForEachEra(_addSeedToTrackProducers, args=["_seedProducers"], names="_seedSelectors", task="_tracksValidationSeedSelectorsTrackingOnly", includeFastSim=True, modDict=globals())
_taskForEachEra(_addSeedToTrackProducers, args=["_seedProducersPreSplitting"], names="_seedSelectorsPreSplitting", task="_tracksValidationSeedSelectorsPreSplittingTrackingOnly", modDict=globals())
tracksValidationSeedSelectorsTrackingOnly.add(tracksValidationSeedSelectorsPreSplittingTrackingOnly)

# MTV instances
trackValidatorTrackingOnly = trackValidatorStandalone.clone(
    label = [ x for x in trackValidatorStandalone.label if x != "cutsRecoTracksAK4PFJets"],
    cores = "highPtJetsForTrk"
 )

trackValidatorSeedingTrackingOnly = _trackValidatorSeedingBuilding.clone(
    dirName = "Tracking/TrackSeeding/",
    label = _seedSelectors,
    doSeedPlots = True,
    doResolutionPlotsForLabels = [ "seedTracksjetCoreRegionalStepSeeds" ]
)
seedingDeepCore.toModify(trackValidatorSeedingTrackingOnly, doResolutionPlotsForLabels = ["seedTracksjetCoreRegionalStepSeedsBarrel","seedTracksjetCoreRegionalStepSeedsEndcap"] )

trackValidatorSeedingPreSplittingTrackingOnly = trackValidatorSeedingTrackingOnly.clone(
    associators = ["quickTrackAssociatorByHitsPreSplitting"],
    label = _seedSelectorsPreSplitting,
    doSummaryPlots = False,

)

trackValidatorJetCoreSeedingTrackingOnly = trackValidatorSeedingTrackingOnly.clone(
    dirName = "Tracking/JetCore/TrackSeeding/",
    associators = ["trackAssociatorByChi2"],
    UseAssociators = True,
    doSeedPlots = True,
)

for _eraName, _postfix, _era in _relevantEras:
    if 'jetCoreRegionalStep' in _cfg.iterationAlgos(_postfix) :
      _setForEra(trackValidatorJetCoreSeedingTrackingOnly, _eraName, _era,
                 label = [ "seedTracksjetCoreRegionalStepSeedsBarrel","seedTracksjetCoreRegionalStepSeedsEndcap" ],
                 doResolutionPlotsForLabels = [ "seedTracksjetCoreRegionalStepSeedsBarrel","seedTracksjetCoreRegionalStepSeedsEndcap" ]
      )
    
for _eraName, _postfix, _era in _relevantErasAndFastSim:
    _setForEra(trackValidatorSeedingTrackingOnly, _eraName, _era, label = locals()["_seedSelectors"+_postfix])
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorSeedingPreSplittingTrackingOnly, _eraName, _era, label = locals()["_seedSelectorsPreSplitting"+_postfix])


trackValidatorConversionTrackingOnly = trackValidatorConversion.clone(label = [x for x in trackValidatorConversion.label if x not in ["ckfInOutTracksFromConversions", "ckfOutInTracksFromConversions"]])

trackValidatorBHadronTrackingOnly = trackValidatorBHadron.clone(label = [x for x in trackValidatorTrackingOnly.label if "Pt09" not in x])

trackValidatorTPPtLess09TrackingOnly = trackValidatorTPPtLess09Standalone.clone(cores = "highPtJetsForTrk")
trackValidatorFromPVTrackingOnly = trackValidatorFromPVStandalone.clone(cores = "highPtJetsForTrk")
trackValidatorFromPVAllTPTrackingOnly = trackValidatorFromPVAllTPStandalone.clone(cores = "highPtJetsForTrk")
trackValidatorAllTPEfficTrackingOnly = trackValidatorAllTPEfficStandalone.clone(cores = "highPtJetsForTrk")
# sequences
tracksPreValidationTrackingOnly = tracksPreValidationStandalone.copy()
tracksPreValidationTrackingOnly.replace(tracksValidationSelectors, tracksValidationSelectorsTrackingOnly)
tracksPreValidationTrackingOnly.replace(highPtJets,highPtJetsForTrk)

trackValidatorsTrackingOnly = _trackValidatorsBase.copy()
trackValidatorsTrackingOnly.replace(trackValidatorStandalone, trackValidatorTrackingOnly)
trackValidatorsTrackingOnly.replace(trackValidatorTPPtLess09Standalone,trackValidatorTPPtLess09TrackingOnly)
trackValidatorsTrackingOnly.replace(trackValidatorFromPVStandalone,trackValidatorFromPVTrackingOnly)
trackValidatorsTrackingOnly.replace(trackValidatorFromPVAllTPStandalone,trackValidatorFromPVAllTPTrackingOnly)
trackValidatorsTrackingOnly.replace(trackValidatorAllTPEfficStandalone,trackValidatorAllTPEfficTrackingOnly)
trackValidatorsTrackingOnly += trackValidatorSeedingTrackingOnly
trackValidatorsTrackingOnly += trackValidatorSeedingPreSplittingTrackingOnly
trackValidatorsTrackingOnly += trackValidatorBuilding
trackValidatorsTrackingOnly += trackValidatorBuildingPreSplitting
trackValidatorsTrackingOnly.replace(trackValidatorConversionStandalone, trackValidatorConversionTrackingOnly)
trackValidatorsTrackingOnly.remove(trackValidatorGsfTracksStandalone)
trackValidatorsTrackingOnly.replace(trackValidatorBHadronStandalone, trackValidatorBHadronTrackingOnly)

seedingDeepCore.toReplaceWith(trackValidatorsTrackingOnly, cms.Sequence(
            trackValidatorsTrackingOnly.copy()+
            trackValidatorJetCore+
            trackValidatorJetCoreSeedingTrackingOnly
            ) 
        )
phase2_tracker.toReplaceWith(trackValidatorsTrackingOnly, trackValidatorsTrackingOnly.copyAndExclude([ #must be done for each era which does not have jetcore in the iteration
    trackValidatorJetCore,
    trackValidatorJetCoreSeedingTrackingOnly
])) 

displacedTrackValidation.toReplaceWith(trackValidatorsTrackingOnly, cms.Sequence(trackValidatorsTrackingOnly.copy()+trackValidatorDisplaced))

fastSim.toReplaceWith(trackValidatorsTrackingOnly, trackValidatorsTrackingOnly.copyAndExclude([
    trackValidatorBuildingPreSplitting,
    trackValidatorSeedingPreSplittingTrackingOnly,
    trackValidatorConversionTrackingOnly,
    trackValidatorBHadronTrackingOnly
]))
tracksValidationTrackingOnly = cms.Sequence(
    trackValidatorsTrackingOnly,
    tracksPreValidationTrackingOnly,
    tracksValidationSelectorsStandalone,
    tracksValidationSeedSelectorsTrackingOnly
)

####################################################################################################
### Pixel tracking only mode (placeholder for now)
trackingParticlePixelTrackAsssociation = trackingParticleRecoTrackAsssociation.clone(
    label_tr = "pixelTracks",
    associator = "quickTrackAssociatorByHitsPreSplitting",
)
PixelVertexAssociatorByPositionAndTracks = VertexAssociatorByPositionAndTracks.clone(
    trackAssociation = "trackingParticlePixelTrackAsssociation"
)

_pixelTracksCustom = dict(
    src = "pixelTracks",
    vertexTag = "pixelVertices",
)

trackRefSelector = cms.EDFilter('TrackRefSelector',
                                src = cms.InputTag('pixelTracks'),
                                cut = cms.string("")
)

trackSelector = cms.EDFilter('TrackSelector',
                             src = cms.InputTag('pixelTracks'),
                             cut = cms.string("")
)

cutstring = "numberOfValidHits == 3"
pixelTracks3hits = trackRefSelector.clone( cut = cutstring )

cutstring = "numberOfValidHits >= 4"
pixelTracks4hits = trackRefSelector.clone( cut = cutstring )

cutstring = "pt > 0.9"
pixelTracksPt09 = trackRefSelector.clone( cut = cutstring )
#pixelTracksPt09 = generalTracksPt09.clone(quality = ["undefQuality"], **_pixelTracksCustom)

pixelTracksFromPV = generalTracksFromPV.clone(quality = "highPurity", ptMin = 0.0, ptMax = 99999., ptErrorCut = 99999., copyExtras = True, **_pixelTracksCustom)
#pixelTracksFromPVPt09 = generalTracksPt09.clone(quality = ["loose","tight","highPurity"], vertexTag = "pixelVertices", src = "pixelTracksFromPV")
pixelTracksFromPVPt09 = pixelTracksFromPV.clone(ptMin = 0.9)

cutstring = "numberOfValidHits >= 4"
#pixelTracksFromPV4hits = trackRefSelector.clone( cut = cutstring, src = "pixelTracksFromPV" )
pixelTracksFromPV4hits = pixelTracksFromPV.clone( numberOfValidPixelHits = 4 )


trackValidatorPixelTrackingOnly = trackValidator.clone(
    dirName = "Tracking/PixelTrack/",
    label = [
        "pixelTracks", "pixelTracksPt09", "pixelTracks3hits", "pixelTracks4hits",
        "pixelTracksL", "pixelTracksPt09L", "pixelTracks3hitsL", "pixelTracks4hitsL",
        "pixelTracksT", "pixelTracksPt09T", "pixelTracks3hitsT", "pixelTracks4hitsT",
        "pixelTracksHP", "pixelTracksPt09HP", "pixelTracks3hitsHP", "pixelTracks4hitsHP",
    ],
    doResolutionPlotsForLabels = [],
    trackCollectionForDrCalculation = "pixelTracks",
    associators = ["trackingParticlePixelTrackAsssociation"],
    label_vertex = "pixelVertices",
    vertexAssociator = "PixelVertexAssociatorByPositionAndTracks",
    dodEdxPlots = False,
    cores = cms.InputTag(""),
)
trackValidatorFromPVPixelTrackingOnly = trackValidatorPixelTrackingOnly.clone(
    dirName = "Tracking/PixelTrackFromPV/",
    label = [
        "pixelTracksFromPV", "pixelTracksFromPVPt09", "pixelTracksFromPV4hits",
        "pixelTracksFromPVL", "pixelTracksFromPVT", "pixelTracksFromPVHP",
        "pixelTracksFromPVPt09L", "pixelTracksFromPVPt09T", "pixelTracksFromPVPt09HP",
        "pixelTracksFromPV4hitsL", "pixelTracksFromPV4hitsT", "pixelTracksFromPV4hitsHP",
    ],
    label_tp_effic = "trackingParticlesSignal",
    label_tp_fake = "trackingParticlesSignal",
    label_tp_effic_refvector = True,
    label_tp_fake_refvector = True,
    trackCollectionForDrCalculation = "pixelTracksFromPV",
    doPlotsOnlyForTruePV = True,
    doPVAssociationPlots = False,
    doResolutionPlotsForLabels = ["disabled"],
)
trackValidatorFromPVAllTPPixelTrackingOnly = trackValidatorFromPVPixelTrackingOnly.clone(
    dirName = "Tracking/PixelTrackFromPVAllTP/",
    label_tp_effic = trackValidatorPixelTrackingOnly.label_tp_effic.value(),
    label_tp_fake = trackValidatorPixelTrackingOnly.label_tp_fake.value(),
    label_tp_effic_refvector = False,
    label_tp_fake_refvector = False,
    doSimPlots = False,
    doSimTrackPlots = False,
)
trackValidatorBHadronPixelTrackingOnly = trackValidatorPixelTrackingOnly.clone(
    dirName = "Tracking/PixelTrackBHadron/",
    label = [
        "pixelTracks", "pixelTracksPt09",
        "pixelTracksL", "pixelTracks3hitsL", "pixelTracks4hitsL",
        "pixelTracksT", "pixelTracks3hitsT", "pixelTracks4hitsT",
        "pixelTracksHP", "pixelTracks3hitsHP", "pixelTracks4hitsHP",
         ],
    label_tp_effic = "trackingParticlesBHadron",
    label_tp_effic_refvector = True,
    doSimPlots = True,
    doRecoTrackPlots = False, # Fake rate is defined wrt. all TPs, and that is already included in trackValidator
    dodEdxPlots = False,
)

tracksValidationTruthPixelTrackingOnly = tracksValidationTruth.copy()
tracksValidationTruthPixelTrackingOnly.replace(trackingParticleRecoTrackAsssociation, trackingParticlePixelTrackAsssociation)
tracksValidationTruthPixelTrackingOnly.replace(VertexAssociatorByPositionAndTracks, PixelVertexAssociatorByPositionAndTracks)
tracksValidationTruthPixelTrackingOnly.add(trackingParticlesBHadron)
tracksValidationTruthPixelTrackingOnly.add( pixelTracks3hits )
tracksValidationTruthPixelTrackingOnly.add( pixelTracks4hits )
tracksValidationTruthPixelTrackingOnly.add( pixelTracksPt09 )
tracksValidationTruthPixelTrackingOnly.add( pixelTracksFromPV )
tracksValidationTruthPixelTrackingOnly.add( pixelTracksFromPVPt09 )
tracksValidationTruthPixelTrackingOnly.add( pixelTracksFromPV4hits )

tracksPreValidationPixelTrackingOnly = cms.Task(
    tracksValidationTruthPixelTrackingOnly,
    trackingParticlesSignal)

##https://cmssdt.cern.ch/lxr/source/DataFormats/TrackReco/interface/TrackBase.h#0150
quality = {
    "L"  : (1,"loose",     ["loose","tight","highPurity"]),
    "T"  : (2,"tight",     ["tight","highPurity"]),
    "HP" : (4,"highPurity",["highPurity"]),
}

for key,value in quality.items():
    qualityName = value[1]
    qualityBit  = value[0]
    qualityList = value[2]

    label = "pixelTracks"+str(key)
    cutstring = "qualityMask <= 7 & qualityMask >= " + str(qualityBit)
    locals()[label] = trackRefSelector.clone( cut = cutstring )
    tracksPreValidationPixelTrackingOnly.add(locals()[label])

    label = "pixelTracksFromPV"+key
#    locals()[label] = generalTracksPt09.clone( ptMin = 0.0, vertexTag = "pixelVertices", src = "pixelTracksFromPV", quality = qualityList )
    locals()[label] = pixelTracksFromPV.clone( quality = qualityName )
    tracksPreValidationPixelTrackingOnly.add(locals()[label])
#-----------    
    cutstring = "pt > 0.9 & qualityMask <= 7 & qualityMask >= " + str(qualityBit) 
    label = "pixelTracksPt09"+key
    locals()[label] = trackRefSelector.clone( cut = cutstring )
    tracksPreValidationPixelTrackingOnly.add(locals()[label])

    label = "pixelTracksFromPVPt09"+key
 #   locals()[label] = generalTracksPt09.clone( ptMin = 0.9, vertexTag = "pixelVertices", src = "pixelTracksFromPV", quality = qualityList )
    locals()[label] = pixelTracksFromPVPt09.clone( quality = qualityName )
    tracksPreValidationPixelTrackingOnly.add(locals()[label])
#-----------         
    label = "pixelTracks4hits"+key
    cutstring = "numberOfValidHits == 4 & qualityMask <= 7 & qualityMask >= " + str(qualityBit)
    locals()[label] = trackRefSelector.clone( cut = cutstring )
    tracksPreValidationPixelTrackingOnly.add(locals()[label])
    
    label = "pixelTracksFromPV4hits"+key
#    locals()[label] = generalTracksPt09.clone( ptMin = 0.0, minHit = 4, vertexTag = "pixelVertices", src = "pixelTracksFromPV", quality = qualityList )
    locals()[label] = pixelTracksFromPV4hits.clone( quality = qualityName )
    tracksPreValidationPixelTrackingOnly.add(locals()[label])
#--------    
    label = "pixelTracks3hits"+key
    cutstring = "numberOfValidHits == 3 & qualityMask <= 7 & qualityMask >= " + str(qualityBit)
    locals()[label] = trackRefSelector.clone( cut = cutstring )
    tracksPreValidationPixelTrackingOnly.add(locals()[label])
     
tracksValidationPixelTrackingOnly = cms.Sequence(
    trackValidatorPixelTrackingOnly +
    trackValidatorFromPVPixelTrackingOnly +
    trackValidatorFromPVAllTPPixelTrackingOnly +
    trackValidatorBHadronPixelTrackingOnly,
    tracksPreValidationPixelTrackingOnly
)
####################################################################################################

### Lite mode (only generalTracks and HP)
trackValidatorLite = trackValidator.clone(
    label = ["generalTracks", "cutsRecoTracksHp"]
)
tracksValidationLite = cms.Sequence(
    cutsRecoTracksHp +
    trackValidatorLite,
    tracksValidationTruth
)

## customization for timing
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify( generalTracksFromPV, 
                              timesTag  = cms.InputTag('tofPID:t0'), 
                              timeResosTag = cms.InputTag('tofPID:sigmat0'),
                              nSigmaDtVertex = cms.double(3) )
phase2_timing_layer.toModify( trackValidatorStandalone,
                              label_vertex = cms.untracked.InputTag('offlinePrimaryVertices4D') )
phase2_timing_layer.toModify( trackValidatorFromPVStandalone,
                              label_vertex = cms.untracked.InputTag('offlinePrimaryVertices4D') )
phase2_timing_layer.toModify( trackValidatorFromPVAllTPStandalone,
                              label_vertex = cms.untracked.InputTag('offlinePrimaryVertices4D') )
phase2_timing_layer.toModify( trackValidatorConversionStandalone,
                              label_vertex = cms.untracked.InputTag('offlinePrimaryVertices4D') )
phase2_timing_layer.toModify( trackValidatorGsfTracks,
                              label_vertex = cms.untracked.InputTag('offlinePrimaryVertices4D') )
