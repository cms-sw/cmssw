import FWCore.ParameterSet.Config as cms
from Validation.Configuration.hltHGCalSimValid_cff import hltRecHitMapProducer

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
    PFCand = cms.InputTag("hltParticleFlow"),
    PFRechit = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    PFCluster = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    CaloParticle = cms.InputTag("mix","MergedCaloTruth"),
    SimCluster = cms.InputTag("mix","MergedCaloTruth"),
    PFClusterSimClusterAssociator = cms.InputTag("hltPFClusterSimClusterAssociationProducerECAL"),
    PFClusterCaloParticleAssociator = cms.InputTag("hltPFClusterCaloParticleAssociationProducerECAL"),
    assocScoreThresholds = cms.vdouble(1.1, 0.9, 0.5, 0.1),
    doMatchByScore = cms.bool(True),
    enFracCut = cms.double(0.),
    ptCut = cms.double(0.)
)

hltPFTesterECALWithCut1 = hltPFTesterECAL.clone(
    enFracCut =  cms.double(0.01),
    ptCut = cms.double(0.)
)

hltPFTesterECALWithCut2 = hltPFTesterECAL.clone(
    enFracCut = cms.double(0.),
    ptCut = cms.double(0.1)
)

hltPFTesterECALWithCut3 = hltPFTesterECAL.clone(
    enFracCut = cms.double(0.01),
    ptCut = cms.double(0.1)
)

# SimToReco match based on shared energy fraction
hltPFTesterECALShEnF = hltPFTesterECAL.clone(
    doMatchByScore = cms.bool(False)
)

hltPFTesterECALShEnFWithCut1 = hltPFTesterECALShEnF.clone(
    enFracCut =  cms.double(0.01),
    ptCut = cms.double(0.)
)

hltPFTesterECALShEnFWithCut2 = hltPFTesterECALShEnF.clone(
    enFracCut = cms.double(0.),
    ptCut = cms.double(0.1)
)

hltPFTesterECALShEnFWithCut3 = hltPFTesterECALShEnF.clone(
    enFracCut = cms.double(0.01),
    ptCut = cms.double(0.1)
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(hltPFTesterECAL, PFCand = cms.InputTag("hltParticleFlowTmp"))

PFValSeq = cms.Sequence(
    hltPFScAssocByEnergyScoreProducer
    +hltPFClusterSimClusterAssociationProducerECAL
    +hltPFCpAssocByEnergyScoreProducer
    +hltPFClusterCaloParticleAssociationProducerECAL
    +hltPFTesterECAL
    +hltPFTesterECALWithCut1
    +hltPFTesterECALWithCut2
    +hltPFTesterECALWithCut3
    +hltPFTesterECALShEnF
    +hltPFTesterECALShEnFWithCut1
    +hltPFTesterECALShEnFWithCut2
    +hltPFTesterECALShEnFWithCut3
)
