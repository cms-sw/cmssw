"""
This config fragment is used for the merging step of the embedding samples.
It merges the collections from the simulation step and the cleaning step.
To do this some producers which are already in the schedule are replaced with embedding merger producers and some are added to the schedule.
The collections which are merged are the collections which serve as input for particle flow producers.
The Simulation RECO step must be carried out beforehand.
To use this config fragment, a cmsDriver command like the following can be used:
```
cmsDriver.py \
	--step USER:TauAnalysis/MCEmbeddingTools/Merging_USER_cff.merge_step,PAT \
	--processName MERGE \
	--data \
	--scenario pp \
	--eventcontent TauEmbeddingMergeMINIAOD \
	--datatier USER \
	--inputCommands 'keep *_*_*_*' \
	--era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Reconstruction_Data_cff import * # most of the producers which are replaced here are imported in this file
from Configuration.StandardSequences.RawToDigi_cff import * # placed this here to be consistent with the older developments, don't know if this is really needed
import PhysicsTools.PatAlgos.tools.coreTools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeMC
from Configuration.ProcessModifiers.tau_embedding_merging_cff import tau_embedding_merging

# overriding behaviour of 'removeMCMatching', as we also use mc and need this so that nanoAODs are correctly produced
# This is needed because we now have a hybrid event which contains both simulation and reconstructed data.
PhysicsTools.PatAlgos.tools.coreTools.removeMCMatching = lambda process, names, postfix, outputModules : miniAOD_customizeMC(process)

from PhysicsTools.PatAlgos.slimming.unpackedPatTrigger_cfi import unpackedPatTrigger

unpackedPatTrigger.triggerResults = cms.InputTag("TriggerResults::SIMembeddingHLT")

# In the following replace producers that produce collections which are used as input for particle flow producers
# by their corresponding merger producers.
# The merger producers merge the collections from the simulation step and the cleaning step.
# Those "toReplaceWith" modifiers need to be applied as late as possible e.g. in StandardSequences.Reconstruction_Data_cff.
# This is to avoid collisions with other modules that are cloned and modified versions from the original modules or "toModify" modifier 
# calls that try to modify the original modules. They throw an exeption trying to do this on the replaced merger modules.

# defined in RecoMuon/MuonIdentification/python/cosmics_id.py
tau_embedding_cosmicsVetoTracksRaw_merger = cms.EDProducer("TrackColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("cosmicsVetoTracksRaw", "", "SIMembedding"),
        cms.InputTag("cosmicsVetoTracksRaw", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(cosmicsVetoTracksRaw, tau_embedding_cosmicsVetoTracksRaw_merger)

# defined in RecoEgamma/EgammaElectronProducers/python/lowPtGsfElectronSequence_cff.py
tau_embedding_lowPtGsfEleGsfTracks_merger = cms.EDProducer("GsfTrackColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("lowPtGsfEleGsfTracks", "", "SIMembedding"),
        cms.InputTag("lowPtGsfEleGsfTracks", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(lowPtGsfEleGsfTracks, tau_embedding_lowPtGsfEleGsfTracks_merger)

# defined in RecoMuon/Configuration/python/MergeDisplacedTrackCollections_cff.py
tau_embedding_displacedTracks_merger = cms.EDProducer("TrackColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("displacedTracks", "", "SIMembedding"),
        cms.InputTag("displacedTracks", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(displacedTracks, tau_embedding_displacedTracks_merger)

# defined in RecoParticleFlow/PFProducer/python/particleFlow_cff.py
tau_embedding_particleFlowTmp_merger = cms.EDProducer("PFColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("particleFlowTmp", "", "SIMembedding"),
        cms.InputTag("particleFlowTmp", "", "LHEembeddingCLEAN"),
        cms.InputTag("particleFlowTmp", "CleanedHF", "SIMembedding"),
        cms.InputTag("particleFlowTmp", "CleanedHF", "LHEembeddingCLEAN"),
        cms.InputTag("particleFlowTmp", "CleanedCosmicsMuons", "SIMembedding"),
        cms.InputTag("particleFlowTmp", "CleanedCosmicsMuons", "LHEembeddingCLEAN"),
        cms.InputTag("particleFlowTmp", "CleanedTrackerAndGlobalMuons", "SIMembedding"),
        cms.InputTag("particleFlowTmp", "CleanedTrackerAndGlobalMuons", "LHEembeddingCLEAN"),
        cms.InputTag("particleFlowTmp", "CleanedFakeMuons", "SIMembedding"),
        cms.InputTag("particleFlowTmp", "CleanedFakeMuons", "LHEembeddingCLEAN"),
        cms.InputTag("particleFlowTmp", "CleanedPunchThroughMuons", "SIMembedding"),
        cms.InputTag("particleFlowTmp", "CleanedPunchThroughMuons", "LHEembeddingCLEAN"),
        cms.InputTag("particleFlowTmp", "CleanedPunchThroughNeutralHadrons", "SIMembedding"),
        cms.InputTag("particleFlowTmp", "CleanedPunchThroughNeutralHadrons", "LHEembeddingCLEAN"),
        cms.InputTag("particleFlowTmp", "AddedMuonsAndHadrons", "SIMembedding"),
        cms.InputTag("particleFlowTmp", "AddedMuonsAndHadrons", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(particleFlowTmp, tau_embedding_particleFlowTmp_merger)

# defined in RecoLocalCalo/EcalRecProducers/python/ecalPreshowerRecHit_cfi.py
tau_embedding_ecalPreshowerRecHit_merger = cms.EDProducer("EcalRecHitColMerger",
    mergeCollections=cms.VInputTag(
        cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES", "SIMembedding"),
        cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES", "LHEembeddingCLEAN"),
    )
)
tau_embedding_merging.toReplaceWith(ecalPreshowerRecHit, tau_embedding_ecalPreshowerRecHit_merger)

# defined in RecoLocalCalo/HcalRecProducers/python/HcalHitReconstructor_ho_cfi.py
tau_embedding_horeco_merger = cms.EDProducer("HORecHitColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("horeco", "", "SIMembedding"),
        cms.InputTag("horeco", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(horeco, tau_embedding_horeco_merger)

# defined in RecoLocalCalo/HcalRecProducers/python/HFPhase1Reconstructor_cfi.py
tau_embedding_hfreco_merger = cms.EDProducer("HFRecHitColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("hfreco", "", "SIMembedding"),
        cms.InputTag("hfreco", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(hfreco, tau_embedding_hfreco_merger)

# defined in RecoTracker/FinalTrackSelectors/python/MergeTrackCollections_cff.py
tau_embedding_generalTracks_merger = cms.EDProducer("TrackColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("generalTracks", "", "SIMembedding"),
        cms.InputTag("generalTracks", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(generalTracks, tau_embedding_generalTracks_merger)

# defined in TrackingTools/GsfTracking/python/GsfElectronGsfFit_cff.py
tau_embedding_electronGsfTracks_merger = cms.EDProducer("GsfTrackColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("electronGsfTracks", "", "SIMembedding"),
        cms.InputTag("electronGsfTracks", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(electronGsfTracks, tau_embedding_electronGsfTracks_merger)

# defined in RecoEgamma/EgammaPhotonProducers/python/allConversions_cfi.py
tau_embedding_allConversions_merger = cms.EDProducer("ConversionColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("allConversions", "", "SIMembedding"),
        cms.InputTag("allConversions", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(allConversions, tau_embedding_allConversions_merger)

# defined in RecoEgamma/EgammaPhotonProducers/python/ckfOutInTracksFromConversions_cfi.py
tau_embedding_ckfOutInTracksFromConversions_merger = cms.EDProducer("TrackColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("ckfOutInTracksFromConversions", "", "SIMembedding"),
        cms.InputTag("ckfOutInTracksFromConversions", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(ckfOutInTracksFromConversions, tau_embedding_ckfOutInTracksFromConversions_merger)

# defined in RecoEgamma/EgammaPhotonProducers/python/conversions_cfi.py
tau_embedding_conversions_merger = cms.EDProducer("ConversionColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("conversions", "", "SIMembedding"),
        cms.InputTag("conversions", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(conversions, tau_embedding_conversions_merger)

# defined in RecoMuon/MuonIdentification/python/muons1stStep_cfi.py
tau_embedding_muons1stStep_merger = cms.EDProducer("MuonColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("muons1stStep", "", "SIMembedding"),
        cms.InputTag("muons1stStep", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(muons1stStep, tau_embedding_muons1stStep_merger)

# defined in RecoMuon/Configuration/python/RecoMuonPPonly_cff.py
tau_embedding_displacedMuons1stStep_merger = cms.EDProducer("MuonColMerger",
    mergeCollections = cms.VInputTag(
        cms.InputTag("displacedMuons1stStep", "", "SIMembedding"),
        cms.InputTag("displacedMuons1stStep", "", "LHEembeddingCLEAN")
    )
)
tau_embedding_merging.toReplaceWith(displacedMuons1stStep, tau_embedding_displacedMuons1stStep_merger)

# defined in RecoLocalCalo/EcalRecProducers/python/ecalRecHit_cfi.py
tau_embedding_ecalRecHit_merger = cms.EDProducer(
    "EcalRecHitColMerger",
    mergeCollections=cms.VInputTag(
        cms.InputTag("ecalRecHit", "EcalRecHitsEB", "SIMembedding"),
        cms.InputTag("ecalRecHit", "EcalRecHitsEB", "LHEembeddingCLEAN"),
        cms.InputTag("ecalRecHit", "EcalRecHitsEE", "SIMembedding"),
        cms.InputTag("ecalRecHit", "EcalRecHitsEE", "LHEembeddingCLEAN"),
    ),
)
tau_embedding_merging.toModify(ecalRecHit, cpu=tau_embedding_ecalRecHit_merger)

# defined in EventFilter/EcalRawToDigi/python/ecalDigis_cfi.py

tau_embedding_ecalDigis_merger = cms.EDProducer(
    "EcalSrFlagColMerger",
    mergeCollections=cms.VInputTag(
        cms.InputTag("ecalDigis", "", "SIMembedding"),
        cms.InputTag("ecalDigis", "", "LHEembeddingCLEAN"),
    ),
)
tau_embedding_merging.toModify(ecalDigis, cpu=tau_embedding_ecalDigis_merger)

# defined in EventFilter/HcalRawToDigi/python/HcalRawToDigi_cfi.py
tau_embedding_hcalDigis_merger = cms.EDProducer(
    "HcalDigiColMerger",
    mergeCollections=cms.VInputTag(
        cms.InputTag("hcalDigis", "", "SIMembedding"),
        cms.InputTag("hcalDigis", "", "LHEembeddingCLEAN"),
    ),
)
tau_embedding_merging.toReplaceWith(hcalDigis, tau_embedding_hcalDigis_merger)

# defined in RecoLocalCalo/Configuration/python/hcalGlobalReco_cff.py
tau_embedding_hbhereco_merger = cms.EDProducer("HBHERecHitColMerger",
    mergeCollections=cms.VInputTag(
        cms.InputTag("hbhereco", "", "SIMembedding"),
        cms.InputTag("hbhereco", "", "LHEembeddingCLEAN"),
    )
)
tau_embedding_merging.toReplaceWith(hbhereco, tau_embedding_hbhereco_merger)

# create a sequence which runs some of the merge producers, which were just created.
merge_step = cms.Sequence(
    ecalDigis
    + hcalDigis
    + generalTracks
    + hbhereco
    + electronGsfTracks
    + ckfOutInTracksFromConversions
    + allConversions
    + muons1stStep
    + displacedMuons1stStep
)
# add more producers which are needed by the PAT step to the sequence
from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import totemRPRawToDigi

totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector", "", "LHC")
merge_step += totemRPRawToDigi

# produce local CT PPS reco
from EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi import ctppsDiamondRawToDigi
from EventFilter.CTPPSRawToDigi.ctppsPixelDigis_cfi import ctppsPixelDigis

ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector", "", "LHC")
ctppsPixelDigis.inputLabel = cms.InputTag("rawDataCollector", "", "LHC")
merge_step += ctppsDiamondRawToDigi + ctppsPixelDigis

from RecoPPS.Configuration.recoCTPPS_cff import recoCTPPSTask

merge_step += cms.Sequence(recoCTPPSTask)

# produce local calo
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import (
    calolocalreco,
    reducedHcalRecHitsSequence,
)

merge_step += calolocalreco + reducedHcalRecHitsSequence

from RecoJets.JetProducers.CaloTowerSchemeB_cfi import towerMaker

# produce hcal towers
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import calotowermaker

merge_step += calotowermaker + towerMaker

# produce clusters
from RecoEcal.Configuration.RecoEcal_cff import ecalClusters

merge_step += ecalClusters

from RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff import (
    particleFlowSuperClusteringSequence,
)

# produce PFCluster Collections
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import (
    particleFlowCluster,
)

merge_step += particleFlowCluster + particleFlowSuperClusteringSequence

# produce muonEcalDetIds
from RecoMuon.MuonIdentification.muons1stStep_cfi import muonEcalDetIds
from RecoMuon.MuonIdentification.muonShowerInformationProducer_cff import (
    muonShowerInformation,
)

merge_step += muonEcalDetIds + muonShowerInformation

# muon Isolation sequences
from RecoMuon.MuonIsolationProducers.muIsolation_cff import (
    muIsolation,
    muIsolationDisplaced,
)

merge_step += muIsolation + muIsolationDisplaced

# muon ID selection type sequences

from RecoMuon.MuonIdentification.muonSelectionTypeValueMapProducer_cff import (
    muonSelectionTypeSequence,
)

merge_step += muonSelectionTypeSequence

from RecoMuon.Configuration.MergeDisplacedTrackCollections_cff import (
    displacedTracksSequence,
)

# displaced muon reduced track extras and tracks
from RecoMuon.MuonIdentification.displacedMuonReducedTrackExtras_cfi import (
    displacedMuonReducedTrackExtras,
)

merge_step += displacedMuonReducedTrackExtras + displacedTracksSequence

# Other things
from RecoTracker.DeDx.dedxEstimators_cff import doAlldEdXEstimators

merge_step += doAlldEdXEstimators

from RecoVertex.Configuration.RecoVertex_cff import (
    unsortedOfflinePrimaryVertices,
    vertexreco,
)

merge_step += vertexreco

unsortedOfflinePrimaryVertices.beamSpotLabel = cms.InputTag(
    "offlineBeamSpot", "", "SELECT"
)
from RecoJets.JetProducers.caloJetsForTrk_cff import ak4CaloJetsForTrk

ak4CaloJetsForTrk.srcPVs = cms.InputTag("firstStepPrimaryVertices", "", "SELECT")

from RecoTracker.DeDx.dedxEstimators_cff import dedxHitInfo

dedxHitInfo.clusterShapeCache = cms.InputTag("")

from Configuration.StandardSequences.Reconstruction_cff import highlevelreco

merge_step += highlevelreco

from CommonTools.ParticleFlow.genForPF2PAT_cff import *

merge_step += genForPF2PATSequence

# total merge_step = cms.Path(ecalDigis+hcalDigis+generalTracks+hbhereco+electronGsfTracks+ckfOutInTracksFromConversions+allConversions+muons1stStep+displacedMuons1stStep+totemRPRawToDigi+ctppsDiamondRawToDigi+ctppsPixelDigis+cms.Sequence(recoCTPPSTask)+calolocalreco+reducedHcalRecHitsSequence+calotowermaker+towerMaker+ecalClusters+particleFlowCluster+particleFlowSuperClusteringSequence+muonEcalDetIds+muonShowerInformation+muIsolation+muIsolationDisplaced+muonSelectionTypeSequence+displacedMuonReducedTrackExtras+displacedTracksSequence+doAlldEdXEstimators+vertexreco+highlevelreco+genForPF2PATSequence)
