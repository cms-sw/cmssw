import FWCore.ParameterSet.Config as cms

_filter_sim_hits = cms.vstring("Ecal",)

hltPFScAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToSCAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:pfRecHitMap"),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection"),
)

hltPFClusterSimClusterAssociationProducerECAL = cms.EDProducer("PCToSCAssociatorEDProducer",
    associator = cms.InputTag("hltPFScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    label_scl = cms.InputTag("mix","MergedCaloTruth"),
    filter_sim_hits = _filter_sim_hits
)

hltPFCpAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToCPAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:pfRecHitMap"),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection"),
)

hltPFClusterCaloParticleAssociationProducerECAL = cms.EDProducer("PCToCPAssociatorEDProducer",
    associator = cms.InputTag("hltPFCpAssocByEnergyScoreProducer"),
    label_lc = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
    filter_sim_hits = _filter_sim_hits
)

hltPFClusterTesterECAL = cms.EDProducer("PFClusterTester",
    PFCand = cms.InputTag("hltParticleFlow"),
    Rechit = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    RecoCluster = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    SimCluster = cms.InputTag("mix","MergedCaloTruth"),
    CaloParticle = cms.InputTag("mix","MergedCaloTruth"),
    ClusterSimClusterAssociator = cms.InputTag("hltPFClusterSimClusterAssociationProducerECAL"),
    ClusterCaloParticleAssociator = cms.InputTag("hltPFClusterCaloParticleAssociationProducerECAL"),
    filter_sim_hits = _filter_sim_hits,
    outFolder = cms.string('HLT/ParticleFlow'),
    assocScoreThresholds = cms.vdouble(1., 0.5, 0.1),
    doMatchByScore = cms.bool(True),
    enFracCut = cms.double(0.),
    ptCut = cms.double(0.),
    etaCut = cms.double(3.0),
)
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(hltPFClusterTesterECAL, PFCand = cms.InputTag("hltParticleFlowTmp"), etaCut = cms.double(1.48))

hltDigisTesterECAL = cms.EDProducer("DigisTester",
    ecalEBDigis = cms.InputTag("hltEcalDigis", "ebDigis"),
    ecalEEDigis = cms.InputTag("hltEcalDigis", "eeDigis"),
    outFolder = cms.string('HLT/ParticleFlow'),
)

hltRecHitTesterECAL = cms.EDProducer("RecHitTester",
    outFolder = cms.string('HLT/ParticleFlow'),
    ebSimHits = cms.InputTag('g4SimHits', 'EcalHitsEB'),
    eeSimHits = cms.InputTag('g4SimHits', 'EcalHitsEE'),
    ebRecHits = cms.InputTag('hltEcalRecHit', 'EcalRecHitsEB'),
    eeRecHits = cms.InputTag('hltEcalRecHit', 'EcalRecHitsEE'),
    ebUncalibRecHits = cms.InputTag('hltEcalUncalibRecHit', 'EcalUncalibRecHitsEB'),
    eeUncalibRecHits = cms.InputTag('hltEcalUncalibRecHit', 'EcalUncalibRecHitsEE'),
    pfRecHits = cms.InputTag('hltParticleFlowRecHitECALUnseeded'),
)
phase2_common.toModify(hltRecHitTesterECAL,
                       eeSimHits = cms.InputTag('g4SimHits', 'HGCHitsEE'),
                       ebUncalibRecHits = cms.InputTag('hltEcalMultiFitUncalibRecHit', 'EcalUncalibRecHitsEB'),
                       eeUncalibRecHits = cms.InputTag('hltEcalMultiFitUncalibRecHit', 'EcalUncalibRecHitsEE')
                       )

hltPFClusterTesterECALWithCut = hltPFClusterTesterECAL.clone(
    enFracCut = cms.double(0.01),
    ptCut = cms.double(0.1)
)

# SimToReco match based on shared energy fraction
hltPFClusterTesterECALShEnF = hltPFClusterTesterECAL.clone(
    doMatchByScore = cms.bool(False)
)

hltPFClusterTesterECALShEnFWithCut = hltPFClusterTesterECALShEnF.clone(
    enFracCut = cms.double(0.01),
    ptCut = cms.double(0.1)
)

PFValSeq = cms.Sequence(
    hltPFScAssocByEnergyScoreProducer
    +hltPFClusterSimClusterAssociationProducerECAL
    +hltPFCpAssocByEnergyScoreProducer
    +hltPFClusterCaloParticleAssociationProducerECAL
    +hltPFClusterTesterECALWithCut
    +hltPFClusterTesterECALShEnFWithCut
    +hltDigisTesterECAL
    +hltRecHitTesterECAL
)
