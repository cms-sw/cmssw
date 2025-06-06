import FWCore.ParameterSet.Config as cms

# from Configuration.ProcessModifiers.tau_embedding_mu_to_mu_cff import tau_embedding_mu_to_mu
# from Configuration.ProcessModifiers.tau_embedding_mu_to_e_cff import tau_embedding_mu_to_e
from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run3_common_cff import run3_common

# from Configuration.StandardSequences.RawToDigi_cff import *
from Configuration.StandardSequences.Reconstruction_Data_cff import *


# overriding behaviour of 'removeMCMatching', as we also use mc and need this so that nanoAODs are correctly produced
import PhysicsTools.PatAlgos.tools.coreTools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeMC

PhysicsTools.PatAlgos.tools.coreTools.removeMCMatching = lambda process, names, postfix, outputModules : miniAOD_customizeMC(process)

from PhysicsTools.PatAlgos.slimming.unpackedPatTrigger_cfi import unpackedPatTrigger
unpackedPatTrigger.triggerResults = cms.InputTag("TriggerResults::SIMembeddingHLT")

generalModifier = run2_common | run3_common


def merge_collections(collection_name, merger_name, import_path, instances=[""]):
    """A function to execute the code to merge the most common collections"""
    # import from the string import_path
    exec(f"from {import_path} import {collection_name} as old_{collection_name}")
    # create the new collection
    exec(
        f'{collection_name} = cms.EDProducer("{merger_name}",\n'
        + f"    mergCollections = cms.VInputTag(\n"
        + ",\n".join(
            [
                f'        cms.InputTag("{collection_name}", "{instance}", "SIMembedding"),\n        cms.InputTag("{collection_name}", "{instance}", "LHEembeddingCLEAN")'
                for instance in instances
            ]
        )
        + "\n    )\n"
        + ")"
    )
    # replace the old collection with the new one
    exec(f"generalModifier.toReplaceWith(old_{collection_name}, {collection_name})")


# ---

merge_collections(
    collection_name="cosmicsVetoTracksRaw",
    merger_name="TrackColMerger",
    import_path="RecoMuon.MuonIdentification.cosmics_id",
)
# ---
merge_collections(
    collection_name="lowPtGsfEleGsfTracks",
    merger_name="GsfTrackColMerger",
    import_path="RecoEgamma.EgammaElectronProducers.lowPtGsfElectronSequence_cff",
)
# ---
merge_collections(
    collection_name="displacedTracks",
    merger_name="TrackColMerger",
    import_path="RecoMuon.Configuration.MergeDisplacedTrackCollections_cff",
)
# ---
merge_collections(
    collection_name="conversions",
    merger_name="ConversionColMerger",
    import_path="RecoEgamma.EgammaPhotonProducers.conversions_cfi",
)

# ---
merge_collections(
    collection_name="particleFlowTmp",
    merger_name="PFColMerger",
    import_path="RecoParticleFlow.PFProducer.particleFlow_cff",
    instances=[
        "",
        "CleanedHF",
        "CleanedCosmicsMuons",
        "CleanedTrackerAndGlobalMuons",
        "CleanedFakeMuons",
        "CleanedPunchThroughMuons",
        "CleanedPunchThroughNeutralHadrons",
        "AddedMuonsAndHadrons",
    ],
)

# ---
# This only worked by replacing the Task and not by replacing the Producer as in the other cases
# because the hbhereco is of the type SwitchProducerCUDA and not of the type EDProducer
# use walrus operator ":=" to give a label to the Producer
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cff import ecalCalibratedRecHitTask

generalModifier.toReplaceWith(
    ecalCalibratedRecHitTask,
    cms.Task(
        ecalRecHit := cms.EDProducer(
            "EcalRecHitColMerger",
            mergCollections=cms.VInputTag(
                cms.InputTag("ecalRecHit", "EcalRecHitsEB", "SIMembedding"),
                cms.InputTag("ecalRecHit", "EcalRecHitsEB", "LHEembeddingCLEAN"),
                cms.InputTag("ecalRecHit", "EcalRecHitsEE", "SIMembedding"),
                cms.InputTag("ecalRecHit", "EcalRecHitsEE", "LHEembeddingCLEAN"),
            ),
        )
    ),
)
# ---
merge_collections(
    collection_name="ecalPreshowerRecHit",
    merger_name="EcalRecHitColMerger",
    import_path="RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi",
    instances=["EcalRecHitsES"],
)


# ---
merge_collections(
    collection_name="horeco",
    merger_name="HORecHitColMerger",
    import_path="RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi",
)
# ---
merge_collections(
    collection_name="hfreco",
    merger_name="HFRecHitColMerger",
    import_path="RecoLocalCalo.Configuration.hcalLocalReco_cff",
)

# (run2_common & ~run3_common).toModify(
#     to_bemanipulate,
#     lambda l: l.extend(
#         [
#             module_manipulate(
#                 module_name="conversionStepTracks",
#                 manipulator_name="Track",
#                 steps=["SIM", "MERGE"],
#             ),
#             module_manipulate(
#                 module_name="ckfInOutTracksFromConversions",
#                 manipulator_name="Track",
#                 steps=["SIM", "MERGE"],
#             ),
#             module_manipulate(
#                 module_name="electronMergedSeeds",
#                 manipulator_name="ElectronSeed",
#                 steps=["SIM", "MERGE"],
#             ),
#             module_manipulate(
#                 module_name="ecalDrivenElectronSeeds",
#                 manipulator_name="EcalDrivenElectronSeed",
#                 steps=["SIM", "MERGE"],
#             ),
#             module_manipulate(module_name="hbheprereco", manipulator_name="HBHERecHit"),
#             module_manipulate(module_name="zdcreco", manipulator_name="ZDCRecHit"),
#         ]
#     ),
# )


# The following collections are also included in the merge_step
# ---

# This only worked by replacing the Task and not by replacing the Producer as in the other cases
# because the hbhereco is of the type SwitchProducerCUDA and not of the type EDProducer
# use walrus operator ":=" to give a label to the Producer
from EventFilter.EcalRawToDigi.ecalDigis_cff import ecalDigisTask

generalModifier.toReplaceWith(
    ecalDigisTask,
    cms.Task(
        ecalDigis := cms.EDProducer(
            "EcalSrFlagColMerger",
            mergCollections=cms.VInputTag(
                cms.InputTag("ecalDigis", "", "SIMembedding"),
                cms.InputTag("ecalDigis", "", "LHEembeddingCLEAN"),
            ),
        )
    ),
)
# ---
from Configuration.StandardSequences.RawToDigi_cff import hcalDigis as old_hcalDigis

hcalDigis = cms.EDProducer(
    "HcalDigiColMerger",
    mergCollections=cms.VInputTag(
        cms.InputTag("hcalDigis", "", "SIMembedding"),
        cms.InputTag("hcalDigis", "", "LHEembeddingCLEAN"),
    ),
)
generalModifier.toReplaceWith(old_hcalDigis, hcalDigis)
# merge_collections(
#     collection_name="hcalDigis",
#     merger_name="HcalDigiColMerger",
#     import_path="Configuration.StandardSequences.RawToDigi_cff",
# )

# ---
merge_collections(
    collection_name="generalTracks",
    merger_name="TrackColMerger",
    import_path="RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff",
)
# ---

from RecoLocalCalo.Configuration.hcalGlobalReco_cff import (
    hcalGlobalRecoTask,
    hcalOnlyGlobalRecoTask,
)

generalModifier.toReplaceWith(
    hcalGlobalRecoTask,
    cms.Task(
        hbhereco := cms.EDProducer(
            "HBHERecHitColMerger",
            mergCollections=cms.VInputTag(
                cms.InputTag("hbhereco", "", "SIMembedding"),
                cms.InputTag("hbhereco", "", "LHEembeddingCLEAN"),
            ),
        )
    ),
)
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB

run3_HB.toReplaceWith(hcalOnlyGlobalRecoTask, cms.Task(hbhereco))

# ---
merge_collections(
    collection_name="electronGsfTracks",
    merger_name="GsfTrackColMerger",
    import_path="TrackingTools.GsfTracking.GsfElectronGsfFit_cff",
)
# ---
merge_collections(
    collection_name="ckfOutInTracksFromConversions",
    merger_name="TrackColMerger",
    import_path="RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi",
)

# ---

merge_collections(
    collection_name="allConversions",
    merger_name="ConversionColMerger",
    import_path="RecoEgamma.EgammaPhotonProducers.allConversions_cfi",
)

# ---

merge_collections(
    collection_name="muons1stStep",
    merger_name="MuonColMerger",
    import_path="RecoMuon.MuonIdentification.muons1stStep_cfi",
)

# ---
merge_collections(
    collection_name="displacedMuons1stStep",
    merger_name="MuonColMerger",
    import_path="RecoMuon.Configuration.RecoMuonPPonly_cff",
)

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
# ---
from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import totemRPRawToDigi

totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector", "", "LHC")
merge_step += totemRPRawToDigi
# ---
# produce local CT PPS reco
from EventFilter.CTPPSRawToDigi.ctppsPixelDigis_cfi import ctppsPixelDigis

# from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import ctppsDiamondRawToDigi
from EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi import ctppsDiamondRawToDigi

ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector", "", "LHC")
ctppsPixelDigis.inputLabel = cms.InputTag("rawDataCollector", "", "LHC")
merge_step += ctppsDiamondRawToDigi + ctppsPixelDigis
# ---
from RecoPPS.Configuration.recoCTPPS_cff import recoCTPPSTask

merge_step += cms.Sequence(recoCTPPSTask)
# ---
# produce local calo
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import (
    calolocalreco,
    reducedHcalRecHitsSequence,
)

merge_step += calolocalreco + reducedHcalRecHitsSequence
# ---
# produce hcal towers
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import calotowermaker
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import towerMaker

merge_step += calotowermaker + towerMaker
# ---
# produce clusters
from RecoEcal.Configuration.RecoEcal_cff import ecalClusters

merge_step += ecalClusters
# ---
# produce PFCluster Collections
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import (
    particleFlowCluster,
)
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff import (
    particleFlowSuperClusteringSequence,
)

merge_step += particleFlowCluster + particleFlowSuperClusteringSequence
# ---
# produce muonEcalDetIds
from RecoMuon.MuonIdentification.muons1stStep_cfi import muonEcalDetIds
from RecoMuon.MuonIdentification.muonShowerInformationProducer_cff import (
    muonShowerInformation,
)

merge_step += muonEcalDetIds + muonShowerInformation
# ---
# muon Isolation sequences
from RecoMuon.MuonIsolationProducers.muIsolation_cff import (
    muIsolation,
    muIsolationDisplaced,
)

merge_step += muIsolation + muIsolationDisplaced
# ---
# muon ID selection type sequences

from RecoMuon.MuonIdentification.muonSelectionTypeValueMapProducer_cff import (
    muonSelectionTypeSequence,
)

merge_step += muonSelectionTypeSequence
# ---
# displaced muon reduced track extras and tracks
from RecoMuon.MuonIdentification.displacedMuonReducedTrackExtras_cfi import (
    displacedMuonReducedTrackExtras,
)
from RecoMuon.Configuration.MergeDisplacedTrackCollections_cff import (
    displacedTracksSequence,
)

merge_step += displacedMuonReducedTrackExtras + displacedTracksSequence
# ---
# Other things
from RecoTracker.DeDx.dedxEstimators_cff import doAlldEdXEstimators

merge_step += doAlldEdXEstimators
from RecoVertex.Configuration.RecoVertex_cff import (
    vertexreco,
    unsortedOfflinePrimaryVertices,
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
# ---
from CommonTools.ParticleFlow.genForPF2PAT_cff import *

merge_step += genForPF2PATSequence

# merge_step = cms.Path(ecalDigis+hcalDigis+generalTracks+hbhereco+electronGsfTracks+ckfOutInTracksFromConversions+allConversions+muons1stStep+displacedMuons1stStep+totemRPRawToDigi+ctppsDiamondRawToDigi+ctppsPixelDigis+cms.Sequence(recoCTPPSTask)+calolocalreco+reducedHcalRecHitsSequence+calotowermaker+towerMaker+ecalClusters+particleFlowCluster+particleFlowSuperClusteringSequence+muonEcalDetIds+muonShowerInformation+muIsolation+muIsolationDisplaced+muonSelectionTypeSequence+displacedMuonReducedTrackExtras+displacedTracksSequence+doAlldEdXEstimators+vertexreco+highlevelreco+genForPF2PATSequence)
