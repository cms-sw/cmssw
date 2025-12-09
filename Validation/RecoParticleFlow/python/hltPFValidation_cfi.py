import FWCore.ParameterSet.Config as cms

hltPFScAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToSCAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:barrelRecHitMap"),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection"),
)

hltPFClusterSimClusterAssociationProducerECAL = cms.EDProducer("PCToSCAssociatorEDProducer",
    associator = cms.InputTag("hltPFScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    label_scl = cms.InputTag("mix","MergedCaloTruth")
)

hltPFCpAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToCPAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:barrelRecHitMap"),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection"),
)

hltPFClusterCaloParticleAssociationProducerECAL = cms.EDProducer("PCToCPAssociatorEDProducer",
    associator = cms.InputTag("hltPFCpAssocByEnergyScoreProducer"),
    label_lc = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    label_cp = cms.InputTag("mix","MergedCaloTruth")
)

hltPFClusterTesterECAL = cms.EDProducer("PFClusterTester",
    PFCand = cms.InputTag("hltParticleFlow"),
    Rechit = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    RecoCluster = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    SimCluster = cms.InputTag("mix","MergedCaloTruth"),
    CaloParticle = cms.InputTag("mix","MergedCaloTruth"),
    ClusterSimClusterAssociator = cms.InputTag("hltPFClusterSimClusterAssociationProducerECAL"),
    ClusterCaloParticleAssociator = cms.InputTag("hltPFClusterCaloParticleAssociationProducerECAL"),
    outFolder = cms.string('HLT/ParticleFlow'),
    assocScoreThresholds = cms.vdouble(1.1, 0.9, 0.5, 0.1),
    doMatchByScore = cms.bool(True),
    enFracCut = cms.double(0.),
    ptCut = cms.double(0.)
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(hltPFClusterTesterECAL, PFCand = cms.InputTag("hltParticleFlowTmp"))

hltPFClusterTesterECALWithCut1 = hltPFClusterTesterECAL.clone(
    enFracCut =  cms.double(0.01),
    ptCut = cms.double(0.)
)

hltPFClusterTesterECALWithCut2 = hltPFClusterTesterECAL.clone(
    enFracCut = cms.double(0.),
    ptCut = cms.double(0.1)
)

hltPFClusterTesterECALWithCut3 = hltPFClusterTesterECAL.clone(
    enFracCut = cms.double(0.01),
    ptCut = cms.double(0.1)
)

# SimToReco match based on shared energy fraction
hltPFClusterTesterECALShEnF = hltPFClusterTesterECAL.clone(
    doMatchByScore = cms.bool(False)
)

hltPFClusterTesterECALShEnFWithCut1 = hltPFClusterTesterECALShEnF.clone(
    enFracCut =  cms.double(0.01),
    ptCut = cms.double(0.)
)

hltPFClusterTesterECALShEnFWithCut2 = hltPFClusterTesterECALShEnF.clone(
    enFracCut = cms.double(0.),
    ptCut = cms.double(0.1)
)

hltPFClusterTesterECALShEnFWithCut3 = hltPFClusterTesterECALShEnF.clone(
    enFracCut = cms.double(0.01),
    ptCut = cms.double(0.1)
)

PFValSeq = cms.Sequence(
    hltPFScAssocByEnergyScoreProducer
    +hltPFClusterSimClusterAssociationProducerECAL
    +hltPFCpAssocByEnergyScoreProducer
    +hltPFClusterCaloParticleAssociationProducerECAL
    +hltPFClusterTesterECAL
    +hltPFClusterTesterECALWithCut1
    +hltPFClusterTesterECALWithCut2
    +hltPFClusterTesterECALWithCut3
    +hltPFClusterTesterECALShEnF
    +hltPFClusterTesterECALShEnFWithCut1
    +hltPFClusterTesterECALShEnFWithCut2
    +hltPFClusterTesterECALShEnFWithCut3
)
