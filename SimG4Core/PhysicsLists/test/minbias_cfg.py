import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD

process = cms.Process('SIM',Run3_DDD)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.Generator.MinBias_14TeV_pythia8_TuneCUETP8M1_cfi')
#process.load('SimG4CMS.Calo.PythiaMinBias_cfi')
process.load('SimG4Core.PhysicsLists.physicsQGSP_FTFP_BERT_G4106_cfi')
#process.load('SimG4Core.PhysicsLists.physicsQGSP_FTFP_BERT_G4104_cfi')
#process.load('SimG4Core.PhysicsLists.physicsQGSP_BERT_G4104_cfi')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.PhysicsList=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)

#process.generator.pythiaHepMCVerbosity = False
#process.generator.pythiaPylistVerbosity = 0
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EMM'

# Schedule definition                                                          
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                )

# filter all path with the production filter sequence                          
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

