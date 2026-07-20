import FWCore.ParameterSet.Config as cms

# Loads LCToSCAssociatorByEnergyScoreProducer, which is a producer that does not make the associations, rather it puts in the event an instance of ticl::LayerClusterToSimClusterAssociatorBaseImplT<reco::CaloCluster (=LayerCluster)>
# in practice an instance of : HGCalLCToSCAssociatorByEnergyScoreImpl = LCToSCAssociatorByEnergyScoreImplT<HGCRecHit, reco::CaloClusterCollection>
# this event data product is then used in LCToSCAssociatorEDProducer to build the actual associations and store them in the event
from SimCalorimetry.HGCalAssociatorProducers.hgCalLCToSCAssociatorByEnergyScoreProducer_cfi import hgCalLCToSCAssociatorByEnergyScoreProducer as scAssocByEnergyScoreProducer
from Configuration.ProcessModifiers.simTrackstersFromPU_cff import simTrackstersFromPU
simTrackstersFromPU.toModify(scAssocByEnergyScoreProducer, hardScatterOnly = cms.bool(False))

from SimCalorimetry.HGCalAssociatorProducers.barrelLCToSCAssociatorByEnergyScoreProducer_cfi import barrelLCToSCAssociatorByEnergyScoreProducer
simTrackstersFromPU.toModify(barrelLCToSCAssociatorByEnergyScoreProducer, hardScatterOnly = cms.bool(False))


from SimCalorimetry.HGCalAssociatorProducers.lcToSCAssociatorEDProducer_cfi import lcToSCAssociatorEDProducer as _lcToSCAssociatorEDProducer
layerClusterSimClusterAssociationProducer = _lcToSCAssociatorEDProducer.clone(
    label_scl = cms.InputTag("mix","MergedCaloTruth"), # will use associator = "scAssocByEnergyScoreProducer"
)
layerClusterBoundaryTrackSimClusterAssociationProducer = _lcToSCAssociatorEDProducer.clone(
    label_scl = cms.InputTag("mix","MergedCaloTruthBoundaryTrackSimCluster"),
)
layerClusterMergedSimClusterAssociationProducer = _lcToSCAssociatorEDProducer.clone(
    label_scl = cms.InputTag("mix","MergedCaloTruthMergedSimCluster"),
)
# the next associator is an associator of LCs->SimCluster dataforamt but using SimCluster collection that is a 1-1 mapping to CaloParticle. 
# this way downstream code only has one dataformat (SimCluster) instead of 2 (CaloParticle & SimCluster)
# at some point layerClusterCaloParticleAssociationProducer will be removed, keeping only layerClusterBoundaryTrackSimClusterAssociationProducer (once downstream code is updated)
layerClusterCaloParticleSimClusterAssociationProducer = _lcToSCAssociatorEDProducer.clone(
    label_scl = cms.InputTag("mix","MergedCaloTruthCaloParticle"),
)

barrelLayerClusterSimClusterAssociation = _lcToSCAssociatorEDProducer.clone(
    associator = cms.InputTag('barrelLCToSCAssociatorByEnergyScoreProducer'),
    label_scl = cms.InputTag("mix","MergedCaloTruth"),
    label_lcl = cms.InputTag("hgcalMergeLayerClusters")
)
barrelLayerClusterCaloParticleAssociation = barrelLayerClusterSimClusterAssociation.clone(
    label_scl = cms.InputTag("mix","MergedCaloTruthCaloParticle"),
)

layerClusterSimClusterAssociationProducerHFNose = layerClusterSimClusterAssociationProducer.clone(
    label_lcl = "hgcalLayerClustersHFNose"
)
layerClusterCaloParticleSimClusterAssociationProducerHFNose = layerClusterCaloParticleSimClusterAssociationProducer.clone(
    label_lcl = "hgcalLayerClustersHFNose"
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
for assoc_ in [layerClusterSimClusterAssociationProducer, layerClusterBoundaryTrackSimClusterAssociationProducer, layerClusterMergedSimClusterAssociationProducer, layerClusterCaloParticleSimClusterAssociationProducer, barrelLayerClusterSimClusterAssociation, layerClusterSimClusterAssociationProducerHFNose, layerClusterCaloParticleSimClusterAssociationProducerHFNose]:
    premix_stage2.toModify(assoc_,
        label_scl = cms.InputTag("mixData", assoc_.label_scl.productInstanceLabel)
    )


