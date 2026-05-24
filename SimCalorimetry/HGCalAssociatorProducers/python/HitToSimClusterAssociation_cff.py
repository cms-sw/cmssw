import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.hitToSimClusterCaloParticleAssociator_cfi import hitToSimClusterCaloParticleAssociator as _hitToSimClusterCaloParticleAssociator


hitToLegacySimClusterAssociator = _hitToSimClusterCaloParticleAssociator.clone(
    simClusters = cms.InputTag("mix", "MergedCaloTruth")
)
hitToBoundarySimClusterAssociator = _hitToSimClusterCaloParticleAssociator.clone(
    simClusters = cms.InputTag("mix", "MergedCaloTruthBoundaryTrackSimCluster")
)
hitToMergedSimClusterAssociator = _hitToSimClusterCaloParticleAssociator.clone(
    simClusters = cms.InputTag("mix", "MergedCaloTruthMergedSimCluster")
)
hitToCPSimClusterAssociator = _hitToSimClusterCaloParticleAssociator.clone(
    simClusters = cms.InputTag("mix", "MergedCaloTruthCaloParticle") # CaloParticle but in SimCluster dataformat
)


from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
for _assoc in [hitToLegacySimClusterAssociator, hitToBoundarySimClusterAssociator, hitToMergedSimClusterAssociator, hitToCPSimClusterAssociator]:
    premix_stage2.toModify(_assoc, simClusters = cms.InputTag("mixData", _assoc.simClusters.productInstanceLabel))

from SimCalorimetry.HGCalAssociatorProducers.barrelHitToSimClusterCaloParticleAssociator_cfi import barrelHitToSimClusterCaloParticleAssociator as _barrelHitToSimClusterCaloParticleAssociator
barrelHitToBoundarySimClusterAssociator = _barrelHitToSimClusterCaloParticleAssociator.clone(
    simClusters = cms.InputTag("mix", "MergedCaloTruthBoundaryTrackSimCluster")
)
barrelHitToCPSimClusterAssociator = _barrelHitToSimClusterCaloParticleAssociator.clone(
    simClusters = cms.InputTag("mix", "MergedCaloTruthCaloParticle") # CaloParticle but in SimCluster dataformat
)
