###############################################################################
# Way to use this:
#   cmsRun testHGCalSingleMuonPt100_cfg.py geometry=D92 type=DDD
#
#   Options for geometry: D88, D92, D93, V17Shift, V18
#               type: DDD, DD4hep
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
                  "geometry of operations: D88, D92, D93, V17Shift, V18")
options.register('type',
                 "DDD",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: DDD, DD4hep")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
if (options.type == "DD4hep"):
    from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
    process = cms.Process('SingleMuonSim',Phase2C17I13M9,dd4hep)
    if (options.geometry == "V17Shift"):
        geomFile = "Geometry.HGCalCommonData.testHGCal" + options.type + options.geometry + "Reco_cff"
    elif (options.geometry == "V18"):
        geomFile = "Geometry.HGCalCommonData.testHGCal" + options.type + options.geometry + "Reco_cff"
    else:
        geomFile = "Configuration.Geometry.Geometry" + options.type +"Extended2026" + options.geometry + "Reco_cff"
else:
    process = cms.Process('SingleMuonSim',Phase2C17I13M9)
    if (options.geometry == "V17Shift"):
        geomFile = "Geometry.HGCalCommonData.testHGCal" + options.geometry + "Reco_cff"
    elif (options.geometry == "V18"):
        geomFile = "Geometry.HGCalCommonData.testHGCal" + options.geometry + "Reco_cff"
    else:
        geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"

globalTag = "auto:phase2_realistic_T21"
outFile = "file:step1" + options.type + options.geometry + "mu.root"

print("Geometry file: ", geomFile)
print("Global Tag:    ", globalTag)
print("Output file:   ", outFile)

# import of standard configurations
process.load(geomFile)
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
process.load('SimG4CMS.Calo.hgcalHitPartial_cff')
process.load("IOMC.RandomEngine.IOMC_cff")

rndm = random.randint(0,200000)
process.RandomNumberGeneratorService.generator.initialSeed = rndm
print("Processing with random number seed: ", rndm)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 1
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalError=dict()
#   process.MessageLogger.HGCSim=dict()
#   process.MessageLogger.HGCalSim=dict()

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
    fileName = cms.untracked.string(outFile),
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
        MinEta = cms.double(2.8),
        MinPhi = cms.double(-3.14159265359) ## in radians
        ),
        Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
        psethack = cms.string('single mu pt 100'),
        firstRun = cms.untracked.uint32(1),
        PythiaParameters = cms.PSet(parameterSets = cms.vstring())
)


#Modified to produce hgceedigis
process.ProductionFilterSequence = cms.Sequence(process.generator)

process.g4SimHits.HGCSD.CheckID = True
process.g4SimHits.HGCScintSD.CheckID = True

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.analysis_step = cms.Path(process.hgcalHitPartialEE+process.hgcalHitPartialHE+process.hgcalHitPartialHEB)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.genfiltersummary_step,
				process.simulation_step,
                                process.endjob_step,
                                process.analysis_step,
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
