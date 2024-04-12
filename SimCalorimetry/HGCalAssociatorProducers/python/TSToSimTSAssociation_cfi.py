import FWCore.ParameterSet.Config as cms

tracksterSimTracksterAssociationLinking = cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
    associator = cms.InputTag('simTracksterHitLCAssociatorByEnergyScoreProducer'),
    label_tst = cms.InputTag("ticlTrackstersMerge"),
    label_simTst = cms.InputTag("ticlSimTracksters", "fromCPs"),
    label_lcl = cms.InputTag("hgcalMergeLayerClusters"),
    label_scl = cms.InputTag("mix", "MergedCaloTruth"),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
)

tracksterSimTracksterAssociationPR = cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
    associator = cms.InputTag('simTracksterHitLCAssociatorByEnergyScoreProducer'),
    label_tst = cms.InputTag("ticlTrackstersMerge"),
    label_simTst = cms.InputTag("ticlSimTracksters"),
    label_lcl = cms.InputTag("hgcalMergeLayerClusters"),
    label_scl = cms.InputTag("mix", "MergedCaloTruth"),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
)


tracksterSimTracksterAssociationLinkingbyCLUE3D = cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
    associator = cms.InputTag('simTracksterHitLCAssociatorByEnergyScoreProducer'),
    label_tst = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    label_simTst = cms.InputTag("ticlSimTracksters", "fromCPs"),
    label_lcl = cms.InputTag("hgcalMergeLayerClusters"),
    label_scl = cms.InputTag("mix", "MergedCaloTruth"),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
)

tracksterSimTracksterAssociationPRbyCLUE3D = cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
    associator = cms.InputTag('simTracksterHitLCAssociatorByEnergyScoreProducer'),
    label_tst = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    label_simTst = cms.InputTag("ticlSimTracksters"),
    label_lcl = cms.InputTag("hgcalMergeLayerClusters"),
    label_scl = cms.InputTag("mix", "MergedCaloTruth"),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
)

tracksterSimTracksterAssociationLinkingPU = cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
    associator = cms.InputTag('simTracksterHitLCAssociatorByEnergyScoreProducer'),
    label_tst = cms.InputTag("ticlTrackstersMerge"),
    label_simTst = cms.InputTag("ticlSimTracksters", "PU"),
    label_lcl = cms.InputTag("hgcalMergeLayerClusters"),
    label_scl = cms.InputTag("mix", "MergedCaloTruth"),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
)

tracksterSimTracksterAssociationPRPU = cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
    associator = cms.InputTag('simTracksterHitLCAssociatorByEnergyScoreProducer'),
    label_tst = cms.InputTag("ticlTrackstersMerge"),
    label_simTst = cms.InputTag("ticlSimTracksters", "PU"),
    label_lcl = cms.InputTag("hgcalMergeLayerClusters"),
    label_scl = cms.InputTag("mix", "MergedCaloTruth"),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(tracksterSimTracksterAssociationLinkingbyCLUE3D, label_tst = cms.InputTag("mergedTrackstersProducer"))
tracksterSimTracksterAssociationLinkingbyCLUE3DEM = tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(label_tst = cms.InputTag("ticlTrackstersCLUE3DEM"))
tracksterSimTracksterAssociationLinkingbyCLUE3DHAD = tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(label_tst = cms.InputTag("ticlTrackstersCLUE3DHAD"))

ticl_v5.toModify(tracksterSimTracksterAssociationPRbyCLUE3D, label_tst = cms.InputTag("mergedTrackstersProducer"))
tracksterSimTracksterAssociationPRbyCLUE3DEM = tracksterSimTracksterAssociationPRbyCLUE3D.clone(label_tst = cms.InputTag("ticlTrackstersCLUE3DEM"))
tracksterSimTracksterAssociationPRbyCLUE3DHAD = tracksterSimTracksterAssociationPRbyCLUE3D.clone(label_tst = cms.InputTag("ticlTrackstersCLUE3DHAD"))

ticl_v5.toModify(tracksterSimTracksterAssociationLinking, label_tst = cms.InputTag("ticlCandidate"))
ticl_v5.toModify(tracksterSimTracksterAssociationPR, label_tst = cms.InputTag("ticlCandidate"))
ticl_v5.toModify(tracksterSimTracksterAssociationLinkingPU, label_tst = cms.InputTag("ticlCandidate"))
ticl_v5.toModify(tracksterSimTracksterAssociationPRPU, label_tst = cms.InputTag("ticlCandidate"))
