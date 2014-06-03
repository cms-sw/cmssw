import FWCore.ParameterSet.Config as cms

process = cms.Process("SIM")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023SHCalNoTaperReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023SHCalNoTaper_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.ShashlikValidation.simhitValidation_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5) )

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(11),
        MinEta = cms.double(1.75),
        MaxEta = cms.double(2.50),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(3.1415926),
        MinE   = cms.double(100.00),
        MaxE   = cms.double(100.00)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

process.options = cms.untracked.PSet(

)

# Output definition

process.ValidationOutput = cms.OutputModule("PoolOutputModule",
                                            outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
                                            fileName = cms.untracked.string('output.root'),
)

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['upgradePLS3']

process.generation_step = cms.Path(process.generator)
process.simulation_step = cms.Path(process.psim)
process.p1 = cms.Path(process.shashlikSimHitValidation)
process.p2 = cms.Path(process.MEtoEDMConverter)
process.output_step = cms.EndPath(process.ValidationOutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.simulation_step,process.p1,process.p2,process.output_step)
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.shashlikSimHitValidation.Verbosity     = 1
