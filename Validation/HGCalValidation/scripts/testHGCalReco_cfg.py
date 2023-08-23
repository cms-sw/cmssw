###############################################################################
# Way to use this:
#   cmsRun testHGCalDigi_cfg.py geometry=D92 type=DDD data=mu tag=Def
#
#   Options for geometry: D88, D92, D93, V17Shift, V18
#               type: DDD, DD4hep
#               data: mu, tt
#               tag: Def, Thr, 0Noise
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
options.register('data',
                 "mu",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "data of operations: mu, tt")
options.register('tag',
                 "Def",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "tag of operations: Def, Thr, 0Noise")
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
inFile = "file:step2" + options.type + options.geometry + options.data + ".root"
outFile = "file:step3" + options.type + options.geometry + options.data  + ".root"
fileName = "missing" + options.type + options.geometry + options.data  + options.tag + ".root"

print("Geometry file: ", geomFile)
print("Global Tag:    ", globalTag)
print("Input file:    ", inFile)
print("Output file:   ", outFile)
print("Root file:     ", fileName)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load(geomFile)
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Validation.HGCalValidation.hgcMissingRecHit_cfi')

process.MessageLogger.cerr.FwkReport.reportEvery = 1
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalMiss=dict()
    process.MessageLogger.HGCalError=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(inFile),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(outFile),
    outputCommands = cms.untracked.vstring(
        'keep *_*hbhe*_*_*',
        'keep *_g4SimHits_*_*',
        'keep *_*HGC*_*_*',
        ),
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True) )

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.recosim_step = cms.Path(process.recosim)
process.analysis_step = cms.Path(process.hgcMissingRecHit)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,
                                process.reconstruction_step,
                                process.recosim_step,
                                process.analysis_step,
                                process.FEVTDEBUGHLToutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
