import FWCore.ParameterSet.Config as cms

_filter_sim_hits = cms.vstring("Ecal",)

_calo_truth = cms.InputTag("mix", "MergedCaloTruth")
_calo_truth_premix = cms.InputTag("mixData", "MergedCaloTruth")

hltPFScAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToSCAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:pfRecHitMap"),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection"),
)

hltPFClusterSimClusterAssociationProducerECAL = cms.EDProducer("PCToSCAssociatorEDProducer",
    associator = cms.InputTag("hltPFScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    label_scl = _calo_truth,
    filter_sim_hits = _filter_sim_hits
)
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hltPFClusterSimClusterAssociationProducerECAL,
    label_scl = _calo_truth_premix,
)

hltPFCpAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToCPAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:pfRecHitMap"),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection"),
)

hltPFClusterCaloParticleAssociationProducerECAL = cms.EDProducer("PCToCPAssociatorEDProducer",
    associator = cms.InputTag("hltPFCpAssocByEnergyScoreProducer"),
    label_lc = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    label_cp = _calo_truth,
    filter_sim_hits = _filter_sim_hits
)
premix_stage2.toModify(hltPFClusterCaloParticleAssociationProducerECAL,
    label_cp = _calo_truth_premix,
)

hltPFClusterTesterECAL = cms.EDProducer("PFClusterTester",
    PFCand = cms.InputTag("hltParticleFlow"),
    Rechit = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    RecoCluster = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
    SimCluster = _calo_truth,
    CaloParticle = _calo_truth,
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
premix_stage2.toModify(hltPFClusterTesterECAL,
    SimCluster = _calo_truth_premix,
    CaloParticle = _calo_truth_premix,
)

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
