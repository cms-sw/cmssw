import FWCore.ParameterSet.Config as cms

process = cms.Process("Sim")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimG4Core.GFlash.TB.h4TB2006GeometryXML_cfi")

process.load("Configuration.EcalTB.simulation_tbsim_cff")

process.load("Configuration.EcalTB.digitization_tbsim_cff")

process.load("Configuration.EcalTB.localReco_tbsim_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        SimEcalTBG4Object = cms.untracked.uint32(5432),
        ecalUnsuppressedDigis = cms.untracked.uint32(54321),
        VtxSmeared = cms.untracked.uint32(12345)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.common_beam_direction_parameters = cms.PSet(
    BeamMeanY = cms.untracked.double(0.0),
    BeamMeanX = cms.untracked.double(0.0),
    MaxEta = cms.untracked.double(0.221525),
    MaxPhi = cms.untracked.double(0.0467617),
    MinEta = cms.untracked.double(0.221525),
    BeamPosition = cms.untracked.double(-26733.5),
    MinPhi = cms.untracked.double(0.0467617)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        process.common_beam_direction_parameters,
        MaxE = cms.untracked.double(20.0),
        MinE = cms.untracked.double(20.0),
        PartID = cms.untracked.vint32(11)
    ),
    Verbosity = cms.untracked.int32(0)
)

process.VtxSmeared = cms.EDProducer("BeamProfileVtxGenerator",
    process.common_beam_direction_parameters,
    BeamSigmaX = cms.untracked.double(2.4),
    BeamSigmaY = cms.untracked.double(2.4),
    GaussianProfile = cms.untracked.bool(False)
)

process.treeProducerCalibSimul = cms.EDAnalyzer("TreeProducerCalibSimul",
    rootfile = cms.untracked.string('treeTB_gf.root'),
    eventHeaderCollection = cms.string(''),
    eventHeaderProducer = cms.string('SimEcalEventHeader'),
    txtfile = cms.untracked.string('treeTB_gf.txt'),
    EBRecHitCollection = cms.string('EcalRecHitsEB'),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    xtalInBeam = cms.untracked.int32(248),
    hodoRecInfoProducer = cms.string('ecalTBSimHodoscopeReconstructor'),
    hodoRecInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
    RecHitProducer = cms.string('ecalTBSimRecHit'),
    tdcRecInfoProducer = cms.string('ecalTBSimTDCReconstructor')
)

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop PSimHits_g4SimHits_*_Sim', 
        'keep PCaloHits_g4SimHits_EcalHitsEB_Sim', 
        'keep PCaloHits_g4SimHits_CaloHitsTk_Sim', 
        'keep PCaloHits_g4SimHits_EcalTBH4BeamHits_Sim'),
    fileName = cms.untracked.string('recoTB_gf.root')
)

process.doSimHits = cms.Sequence(process.VtxSmeared*process.g4SimHits)
process.doSimTB = cms.Sequence(process.SimEcalTBG4Object*process.SimEcalTBHodoscope*process.SimEcalEventHeader)
process.doEcalDigis = cms.Sequence(process.mix*process.ecalUnsuppressedDigis)
process.p1 = cms.Path(process.doSimHits*process.doSimTB*process.doEcalDigis*process.localReco_tbsim*process.treeProducerCalibSimul)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('EcalTBH4Trigger'),
    verbose = cms.untracked.bool(False),
    trigEvents = cms.untracked.int32(100000)
))
process.g4SimHits.Physics.type = 'SimG4Core/Physics/GFlash'
process.g4SimHits.Physics.GFlash = cms.PSet(
    bField = cms.double(0.0),
    GflashEMShowerModel = cms.bool(True),
    GflashHadronShowerModel = cms.bool(True),
    GflashHistogram = cms.bool(True),
    GflashHistogramName = cms.string('gflash_histogram_h4.root'),
    GflashHadronPhysics = cms.string('QGSP_BERT'),
    tuning_pList = cms.vdouble()
)
process.ecal_notCont_sim.EBs25notContainment = 1.0
process.ecal_notCont_sim.EEs25notContainment = 1.0


