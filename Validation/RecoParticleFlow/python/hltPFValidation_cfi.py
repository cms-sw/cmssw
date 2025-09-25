import FWCore.ParameterSet.Config as cms

hltPFScAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToSCAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:barrelRecHitMap"),
    hits = cms.VInputTag("hltParticleFlowRecHitECALUnseeded", "hltParticleFlowRecHitHBHE"), # hltParticleFlowClusterHO
)

hltPFClusterSimClusterAssociationProducerECAL = cms.EDProducer("PCToSCAssociatorEDProducer",
    associator = cms.InputTag("hltPFScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    label_scl = cms.InputTag("mix","MergedCaloTruth")
)

hltPFCpAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToCPAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:barrelRecHitMap"),
    hits = cms.VInputTag("hltParticleFlowRecHitECALUnseeded", "hltParticleFlowRecHitHBHE"), # hltParticleFlowClusterHO
)

hltPFClusterCaloParticleAssociationProducerECAL = cms.EDProducer("PCToCPAssociatorEDProducer",
    associator = cms.InputTag("hltPFCpAssocByEnergyScoreProducer"),
    label_lc = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    label_cp = cms.InputTag("mix","MergedCaloTruth")
)

hltPFTesterECAL = cms.EDProducer("PFTester",
    PFCand = cms.InputTag("hltParticleFlowTmp"),
    PFClusterHCAL = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    SimClusterHCAL = cms.InputTag("mix","MergedCaloTruth"),
    PFClusterSimClusterAssociatorHCAL = cms.InputTag("hltPFClusterSimClusterAssociationProducerECAL"),
    PFClusterCaloParticleAssociatorHCAL = cms.InputTag("hltPFClusterCaloParticleAssociationProducerECAL"),
    assocScoreThreshold = cms.double(0.)
)

PFValSeq = cms.Sequence(
    hltPFScAssocByEnergyScoreProducer
    +hltPFClusterSimClusterAssociationProducerECAL
    +hltPFCpAssocByEnergyScoreProducer
    +hltPFClusterCaloParticleAssociationProducerECAL
    +hltPFTesterECAL
)
