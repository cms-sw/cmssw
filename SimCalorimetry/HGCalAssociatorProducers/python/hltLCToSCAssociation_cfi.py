import FWCore.ParameterSet.Config as cms

### Step 1
# Loads LCToSCAssociatorByEnergyScoreProducer, which is a producer that does not make the associations, rather it puts in the event an instance of ticl::LayerClusterToSimClusterAssociatorBaseImplT<reco::CaloCluster (=LayerCluster)>
# this event data product is then used in LCToSCAssociatorEDProducer to build the actual associations and store them in the event
from SimCalorimetry.HGCalAssociatorProducers.barrelLCToSCAssociatorByEnergyScoreProducer_cfi import barrelLCToSCAssociatorByEnergyScoreProducer as _barrelLCToSCAssociatorByEnergyScoreProducer
# using BarrelLCToSCAssociatorByEnergyScoreProducer = LCToSCAssociatorByEnergyScoreProducerT<reco::PFRecHit, reco::CaloClusterCollection>;
hltBarrelLCToSCAssociatorByEnergyScoreProducer = _barrelLCToSCAssociatorByEnergyScoreProducer.clone(
    hitMapTag = 'hltRecHitMapProducer:barrelRecHitMap',
    hits = 'hltRecHitMapProducer:RefProdVectorPFRecHitCollection'
)

from SimCalorimetry.HGCalAssociatorProducers.hgCalLCToSCAssociatorByEnergyScoreProducer_cfi import hgCalLCToSCAssociatorByEnergyScoreProducer as _hgCalLCToSCAssociatorByEnergyScoreProducer
hltHGCalLCToSCAssociatorByEnergyScoreProducer = _hgCalLCToSCAssociatorByEnergyScoreProducer.clone(
    hitMapTag = 'hltRecHitMapProducer:hgcalRecHitMap',
    hits = 'hltRecHitMapProducer:RefProdVectorHGCRecHitCollection'
)
from Configuration.ProcessModifiers.simTrackstersFromPU_cff import simTrackstersFromPU
simTrackstersFromPU.toModify(hltHGCalLCToSCAssociatorByEnergyScoreProducer, hardScatterOnly = cms.bool(False))

### Step 2: Barrel associators to SimCluster and CaloParticles
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cff import barrelLayerClusterSimClusterAssociation as _barrelLayerClusterSimClusterAssociation
# we use boundary SimCLuster here by default for simplicity
hltBarrelLayerClusterSimClusterAssociation = _barrelLayerClusterSimClusterAssociation.clone(
    associator = 'hltBarrelLCToSCAssociatorByEnergyScoreProducer',
    label_lcl = 'hltMergeLayerClusters',
    label_scl = cms.InputTag("mix","MergedCaloTruthBoundaryTrackSimCluster")
)
hltBarrelLayerClusterCaloParticleAssociation = _barrelLayerClusterSimClusterAssociation.clone(
    associator = 'hltBarrelLCToSCAssociatorByEnergyScoreProducer',
    label_lcl = 'hltMergeLayerClusters',
    label_scl = cms.InputTag("mix","MergedCaloTruthCaloParticle")
)

### Step 3: HGCal associators to SimCluster and CaloParticles
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cff import layerClusterSimClusterAssociationProducer as _layerClusterSimClusterAssociationProducer
hltHGCalLayerClusterSimClusterAssociation = _layerClusterSimClusterAssociationProducer.clone(
    associator = 'hltHGCalLCToSCAssociatorByEnergyScoreProducer',
    label_lcl = 'hltMergeLayerClusters',
    label_scl = cms.InputTag("mix","MergedCaloTruthBoundaryTrackSimCluster")
)
hltHGCalLayerClusterCaloParticleAssociation = _layerClusterSimClusterAssociationProducer.clone(
    associator = 'hltHGCalLCToSCAssociatorByEnergyScoreProducer',
    label_lcl = 'hltMergeLayerClusters',
    label_scl = cms.InputTag("mix","MergedCaloTruthCaloParticle"),
)
