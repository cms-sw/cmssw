# Configuration file to run only TICL on an already existing step3.root file
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9

options = VarParsing.VarParsing ('standard')
options.register('inputFile', 'step3.root', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Input file")
options.register('outputFile', 'step3_reReco_ticlOnly.root', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Output file")
options.register ('globalTag', 'auto:phase2_realistic_T15', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "GlobalTag")
options.parseArguments()

print("Input file: ", options.inputFile)
print("Output file: ", options.outputFile)
print("GlobalTag: ", options.globalTag)

process = cms.Process('TICL',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:'+options.inputFile),
    secondaryFileNames = cms.untracked.vstring()
)
process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
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

process.ticl_FEVToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:'+options.outputFile),
    outputCommands = cms.untracked.vstring( (
        'drop *', 
        'keep *_HGCalRecHit_*_*', 
        'keep recoCaloClusters_hgcalLayerClusters_*_*', 
        'keep *_hgcalLayerClusters_timeLayerCluster_*', 
        'keep *_hgcalLayerClusters_InitialLayerClustersMask_*',
        'keep *_hgcalMultiClusters_*_*', 
        'keep *_iterHGCalMultiClusters_*_*',
        'keep *_ticlTracksters*_*_*', 
        'keep *_ticlSimTracksters*_*_*',
        'keep *_ticlMultiClustersFromTracksters*_*_*', 
        'keep *_ticlMultiClustersFromSimTracksters*_*_*', 
        'keep *_genParticle*_*_*', 
        'keep *_generator_*_*', 
        'keep *_mix*_MergedCaloTruth_*',
        'keep *_genPUProtons_*_*', 
    ) ),
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.ticl_seq = cms.Sequence(
    process.iterTICLTask
)

process.ticl_step = cms.Path(process.ticl_seq)
process.ticl_prevalidation_step = cms.Path(process.globalPrevalidationHGCal)
process.ticl_validation = cms.Sequence(process.hgcalLayerClusters+process.hgcalRecHitMapProducer+process.hgcalValidatorSequence)
process.ticl_validation_step = cms.EndPath(process.ticl_validation)
process.ticl_FEVToutput_step = cms.EndPath(process.ticl_FEVToutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(
    process.ticl_step,
    process.ticl_prevalidation_step,
    process.ticl_validation_step,
    process.ticl_FEVToutput_step,
    process.DQMoutput_step)

# customisation of the process.

# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
