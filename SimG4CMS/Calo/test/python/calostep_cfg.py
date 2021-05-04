import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process("Sim",Run2_2018)

process.load("SimG4CMS.Calo.PythiaMinBias_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Configuration.Geometry.GeometryExtended2018NoSD_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.Step=dict()
    process.MessageLogger.HCalGeom=dict()
    process.MessageLogger.HcalSim=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(-3.0),
        MaxEta = cms.double(3.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinPt  = cms.double(100.),
        MaxPt  = cms.double(100.)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(True)
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    dump = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('singleMuon_FTFP_BERT_EMM.root')
)

# Event output
process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent_singleMuon_FTFP_BERT_EMM.root')
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.analysis_step   = cms.Path(process.CaloSimHitStudy)
process.out_step = cms.EndPath(process.output)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.CaloSimHitStudy.TestNumbering = True

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
        type         = cms.string('CaloSteppingAction'),
        CaloSteppingAction = cms.PSet(
            EBSDNames       = cms.vstring('EBRY'),
            EESDNames       = cms.vstring('EFRY'),
            HCSDNames       = cms.vstring('HBS','HES','HTS'),
            AllSteps        = cms.int32(100),
            SlopeLightYield = cms.double(0.02),
            BirkC1EC        = cms.double(0.03333),
            BirkSlopeEC     = cms.double(0.253694),
            BirkCutEC       = cms.double(0.1),
            BirkC1HC        = cms.double(0.0052),
            BirkC2HC        = cms.double(0.142),
            BirkC3HC        = cms.double(1.75),
            HitCollNames    = cms.vstring('EcalHitsEB1','EcalHitsEE1',
                                          'HcalHits1'),
            EtaTable        = cms.vdouble(0.000, 0.087, 0.174, 0.261, 0.348,
                                          0.435, 0.522, 0.609, 0.696, 0.783,
                                          0.870, 0.957, 1.044, 1.131, 1.218,
                                          1.305, 1.392, 1.479, 1.566, 1.653,
                                          1.740, 1.830, 1.930, 2.043, 2.172,
                                          2.322, 2.500, 2.650, 2.868, 3.000),
            PhiBin         = cms.vdouble(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                         5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                         5.0, 5.0, 5.0, 5.0, 5.0, 5.0,10.0,
                                        10.0,10.0,10.0,10.0,10.0,10.0,10.0,
                                        10.0),
            PhiOffset      = cms.vdouble( 0.0, 0.0, 0.0,10.0),
            EtaMin         = cms.vint32(1, 16, 29, 1),
            EtaMax         = cms.vint32(16, 29, 41, 15),
            EtaHBHE        = cms.int32(16),
            DepthHBHE      = cms.vint32(2,4),
            Depth29Max     = cms.int32(3),
            RMinHO         = cms.double(3800),
            ZHO            = cms.vdouble(0,1255,1418,3928,4100,6610),
            Eta1           = cms.untracked.vint32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                  1, 1, 1, 1, 1, 1, 1, 4, 4),
            Eta15          = cms.untracked.vint32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                  1, 1, 1, 2, 2, 2, 2, 4, 4),
            Eta16          = cms.untracked.vint32(1, 1, 2, 2, 2, 2, 2, 2, 2, 4,
                                                  4, 4, 4, 4, 4, 4, 4, 4, 4),
            Eta17          = cms.untracked.vint32(2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                                  3, 3, 3, 3, 3, 3, 3, 3, 3),
            Eta18          = cms.untracked.vint32(1, 2, 2, 2, 3, 3, 3, 3, 4, 4,
                                                  4, 5, 5, 5, 5, 5, 5, 5, 5),
            Eta19          = cms.untracked.vint32(1, 1, 2, 2, 2, 3, 3, 3, 4, 4,
                                                  4, 5, 5, 5, 5, 6, 6, 6, 6),
            Eta26          = cms.untracked.vint32(1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                                                  5, 6, 6, 6, 6, 7, 7, 7, 7),
            )))

# Schedule definition                                                          
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.analysis_step,
                                process.out_step
                                )

# filter all path with the production filter sequence                          
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

