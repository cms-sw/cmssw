###############################################################################
# Way to use this:
#   cmsRun testHGCalSingleMuonPt100_cfg.py geometry=D92
#
#   Options for geometry D49, D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re, random
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D92",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D49, D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D49"):
    from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
    process = cms.Process('SingleMuon',Phase2C9)
    process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
    globalTag = "auto:phase2_realistic_T15"
elif (options.geometry == "D88"):
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('SingleMuon',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    globalTag = "auto:phase2_realistic_T21"
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('SingleMuon',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D93_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
    globalTag = "auto:phase2_realistic_T21"
else:
    from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
    process = cms.Process('SingleMuon',Phase2C11I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D92_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    globalTag = "auto:phase2_realistic_T21"

print("Global Tag: ", globalTag)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("IOMC.RandomEngine.IOMC_cff")

rndm = random.randint(0,200000)
process.RandomNumberGeneratorService.generator.initialSeed = rndm
print("Processing with random number seed: ", rndm)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.ValidHGCal=dict()
    process.MessageLogger.HGCalGeom=dict()

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string(''),
    annotation = cms.untracked.string(''),
    name = cms.untracked.string('Applications')
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-RECO')
    ),
    fileName = cms.untracked.string('step1.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

process.generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(100.01),
        MinPt = cms.double(99.99),
        ParticleID = cms.vint32(-13),
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(3.1),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(1.3),
        MinPhi = cms.double(-3.14159265359) ## in radians
        ),
        Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
        psethack = cms.string('single mu pt 100'),
        firstRun = cms.untracked.uint32(1),
        PythiaParameters = cms.PSet(parameterSets = cms.vstring())
)


#Modified to produce hgceedigis
process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.genfiltersummary_step,
				process.simulation_step,
                                process.endjob_step,
				process.out_step
				)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path).insert(0, process.ProductionFilterSequence)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
