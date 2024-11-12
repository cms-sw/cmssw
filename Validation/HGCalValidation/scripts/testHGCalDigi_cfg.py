###############################################################################
# Way to use this:
#   cmsRun testHGCalDigi_cfg.py geometry=D111 type=DDD data=mu noise=none
#                               threshold=none
#
#   Options for geometry: D105, D111, D114, V17Shift, D104
#               type: DDD, DD4hep
#               data: mu, tt
#               noise: none, ok
#               threshold: none, ok
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re, random
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D111",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D105, D111, D114, V17Shift, D104")
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
options.register('noise',
                 "ok",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                 "noise of operations: none, ok")
options.register('threshold',
                 "ok",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                 "threshold of operations: none, ok")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
if (options.type == "DD4hep"):
    from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
    if (options.geometry == "V17Shift"):
        from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
        process = cms.Process('SingleMuonSim',Phase2C17I13M9,dd4hep)
        geomFile = "Geometry.HGCalCommonData.testHGCal" + options.type + options.geometry + "Reco_cff"
    elif (options.geometry == "D104"):
        from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
        process = cms.Process('SingleMuonSim',Phase2C22I13M9,dd4hep)
        geomFile = "Configuration.Geometry.Geometry" + options.type +"ExtendedRun4" + options.geometry + "Reco_cff"
    else:
        from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
        process = cms.Process('SingleMuonSim',Phase2C17I13M9,dd4hep)
        geomFile = "Configuration.Geometry.Geometry" + options.type +"ExtendedRun4" + options.geometry + "Reco_cff"
else:
    if (options.geometry == "V17Shift"):
        from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
        process = cms.Process('SingleMuonSim',Phase2C17I13M9)
        geomFile = "Geometry.HGCalCommonData.testHGCal" + options.geometry + "Reco_cff"
    elif (options.geometry == "D104"):
        from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
        process = cms.Process('SingleMuonSim',Phase2C22I13M9)
        geomFile = "Configuration.Geometry.GeometryExtendedRun4" + options.geometry + "Reco_cff"
    else:
        from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
        process = cms.Process('SingleMuonSim',Phase2C17I13M9)
        geomFile = "Configuration.Geometry.GeometryExtendedRun4" + options.geometry + "Reco_cff"

globalTag = "auto:phase2_realistic_T33"
inFile = "file:step1" + options.type + options.geometry + options.data + ".root"
outFile = "file:step2" + options.type + options.geometry + options.data + ".root"

print("Geometry file: ", geomFile)
print("Global Tag:    ", globalTag)
print("Input file:    ", inFile)
print("Output file:   ", outFile)

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hgceeDigitizer
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hgchefrontDigitizer
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hgchebackDigitizer
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hfnoseDigitizer
from SimCalorimetry.HGCalSimProducers.hgcROCParameters_cfi import hgcROCSettings
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load(geomFile)
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_Fake2_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(inFile),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_genParticles_*_*',
        'drop *_genParticlesForJets_*_*',
        'drop *_kt4GenJets_*_*',
        'drop *_kt6GenJets_*_*',
        'drop *_iterativeCone5GenJets_*_*',
        'drop *_ak4GenJets_*_*',
        'drop *_ak7GenJets_*_*',
        'drop *_ak8GenJets_*_*',
        'drop *_ak4GenJetsNoNu_*_*',
        'drop *_ak8GenJetsNoNu_*_*',
        'drop *_genCandidatesForMET_*_*',
        'drop *_genParticlesForMETAllVisible_*_*',
        'drop *_genMetCalo_*_*',
        'drop *_genMetCaloAndNonPrompt_*_*',
        'drop *_genMetTrue_*_*',
        'drop *_genMetIC5GenJs_*_*'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
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
    annotation = cms.untracked.string('step2 nevts:1000'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(outFile),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

if (options.noise == "none"):
    process.HGCAL_noise_fC.values = [0,0,0]

if (options.threshold == "none"):
    process.mix.digitizers.hgceeDigitizer.digiCfg.feCfg.adcThreshold_fC = 0.0
    process.mix.digitizers.hgchefrontDigitizer.digiCfg.feCfg.adcThreshold_fC = 0.0
    process.mix.digitizers.hgchebackDigitizer.digiCfg.feCfg.adcThreshold_fC = 0.0

# Schedule definition
# process.schedule imported from cff in HLTrigger.Configuration
process.schedule.insert(0, process.digitisation_step)
process.schedule.insert(1, process.L1TrackTrigger_step)
process.schedule.insert(2, process.L1simulation_step)
process.schedule.insert(3, process.digi2raw_step)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
