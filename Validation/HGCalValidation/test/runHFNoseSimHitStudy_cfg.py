import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process('PROD',eras.Phase2C6)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load('Configuration.Geometry.GeometryExtended2023D31_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Validation.HGCalValidation.hfnoseSimHitStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase2_realistic']

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalGeom')
    process.MessageLogger.categories.append('HFNSim')
    process.MessageLogger.categories.append('HGCalValidation')

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(3.00),
        MaxEta = cms.double(4.30),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(3.1415926),
        MinE   = cms.double(1000.00),
        MaxE   = cms.double(1000.00)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(True)
)

process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent.root')
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hfnSimHit.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.analysis_step   = cms.Path(process.hgcalSimHitStudy)
process.out_step = cms.EndPath(process.output)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.Physics.DefaultCutValue   = 0.1
process.hgcalSimHitStudy.verbosity = 0

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.analysis_step,
                                )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
