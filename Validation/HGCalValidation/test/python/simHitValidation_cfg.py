import FWCore.ParameterSet.Config as cms
import six

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
process = cms.Process('testHGCalSIMLocal',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D46Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D46_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(1.75),
        MaxEta = cms.double(2.50),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(3.1415926),
        MinE   = cms.double(10.00),
        MaxE   = cms.double(10.00)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(True)
)


# Output definition


process.ValidationOutput = cms.OutputModule("PoolOutputModule",
                                            outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
                                            fileName = cms.untracked.string('file:output.root'),
                                            )

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.HGCalValidation.simhitValidation_cff")

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['upgradePLS3']


# Path and EndPath definitions
process.generation_step = cms.Path(process.generator)
process.simulation_step = cms.Path(process.psim)
process.p1 = cms.Path(process.hgcalSimHitValidationEE+process.hgcalSimHitValidationHEF+process.hgcalSimHitValidationHEB)
process.p2 = cms.Path(process.MEtoEDMConverter)
process.output_step = cms.EndPath(process.ValidationOutput)


# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.simulation_step,process.p1,process.p2,process.output_step)
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.g4SimHits.Physics.DefaultCutValue   = 0.1
process.g4SimHits.HCalSD.TestNumberingScheme = True

for label, prod in six.iteritems(process.producers_()):
        if prod.type_() == "OscarMTProducer":
            # ugly hack
            prod.__dict__['_TypedParameterizable__type'] = "OscarProducer"
