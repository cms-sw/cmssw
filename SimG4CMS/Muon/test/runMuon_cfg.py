import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD

process = cms.Process('SIM',Run3_DDD)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load('Geometry.CMSCommonData.cmsExtendedGeometry2021XML_cfi')
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
process.load('Geometry.EcalCommonData.ecalSimulationParameters_cff')
process.load('Geometry.HcalCommonData.hcalDDDSimConstants_cff')
process.load('Geometry.MuonNumbering.muonGeometryConstants_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.MuonSim=dict()
    process.MessageLogger.MuonGeom=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(-2.4),
        MaxEta = cms.double(2.4),
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

process.genstepfilter.triggerConditions=cms.vstring("generation_step")
process.ProductionFilterSequence = cms.Sequence(process.generator)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
#                                process.genfiltersummary_step,
                                process.simulation_step,
                                process.endjob_step
                                )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path).insert(0, process.ProductionFilterSequence)
