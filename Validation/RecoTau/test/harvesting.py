# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step4 --mc --filetype DQM --conditions auto:phase1_2017_realistic -s HARVESTING:@allForPrompt --era Run2_2017 --scenario pp --filein file:RECO_RAW2DIGI_L1Reco_RECO_EI_PAT_VALIDATION_DQM_inDQM.root --python_filename=harvesting.py --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017

process = cms.Process('HARVESTING',Run2_2017)

process_name = 'QCD'

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:PAT_VALIDATION_DQM_'+process_name+'.root')
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring('ProductNotFound'),
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
    annotation = cms.untracked.string('step4 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

# Path and EndPath definitions
process.alcaHarvesting = cms.Path()
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep+process.DQMOffline_Certification)
process.dqmHarvestingExtraHLT = cms.Path(process.DQMOffline_SecondStep_ExtraHLT+process.DQMOffline_Certification)
process.dqmHarvestingFakeHLT = cms.Path(process.DQMOffline_SecondStep_FakeHLT+process.DQMOffline_Certification)
process.dqmHarvestingPOGMC = cms.Path(process.DQMOffline_SecondStep_PrePOGMC)
process.genHarvesting = cms.Path(process.postValidation_gen)
process.validationHarvesting = cms.Path(process.postValidation+process.hltpostvalidation+process.postValidation_gen)
process.validationHarvestingFS = cms.Path(process.recoMuonPostProcessors+process.postValidationTracking+process.MuIsoValPostProcessor+process.calotowersPostProcessor+process.hcalSimHitsPostProcessor+process.hcaldigisPostProcessor+process.hcalrechitsPostProcessor+process.electronPostValidationSequence+process.photonPostProcessor+process.pfJetClient+process.pfMETClient+process.pfJetResClient+process.pfElectronClient+process.rpcRecHitPostValidation_step+process.makeBetterPlots+process.bTagCollectorSequenceMCbcl+process.METPostProcessor+process.L1GenPostProcessor+process.bdHadronTrackPostProcessor+process.MuonCSCDigisPostProcessors+process.siPixelPhase1OfflineDQM_harvestingV+process.MuonGEMHitsPostProcessors+process.MuonGEMDigisPostProcessors+process.MuonGEMRecHitsPostProcessors+process.postValidation_gen)
process.validationHarvestingHI = cms.Path(process.postValidationHI)
process.validationHarvestingMiniAOD = cms.Path(process.JetPostProcessor+process.METPostProcessorHarvesting+process.bTagMiniValidationHarvesting+process.postValidationMiniAOD)
process.validationHarvestingNoHLT = cms.Path(process.postValidation+process.postValidation_gen)
process.validationpreprodHarvesting = cms.Path(process.postValidation_preprod+process.hltpostvalidation_preprod+process.postValidation_gen)
process.validationpreprodHarvestingNoHLT = cms.Path(process.postValidation_preprod+process.postValidation_gen)
process.validationprodHarvesting = cms.Path(process.hltpostvalidation_prod+process.postValidation_gen)
process.DQMHarvestMuon_step = cms.Path(process.DQMHarvestMuon)
process.DQMCertMuon_step = cms.Path(process.DQMCertMuon)
#process.DQMHarvestL1T_step = cms.Path(process.DQMHarvestL1T)
process.DQMHarvestL1T_step = cms.Path() #MB: empty path as a workaround of unknown source of crashes in tau tests
process.DQMHarvestHcal_step = cms.Path(process.DQMHarvestHcal)
process.DQMHarvestJetMET_step = cms.Path(process.DQMHarvestJetMET)
process.DQMCertJetMET_step = cms.Path(process.DQMCertJetMET)
process.DQMHarvestEcal_step = cms.Path(process.DQMHarvestEcal)
process.DQMCertEcal_step = cms.Path(process.DQMCertEcal)
process.DQMHarvestEGamma_step = cms.Path(process.DQMHarvestEGamma)
process.DQMCertEGamma_step = cms.Path(process.DQMCertEGamma)
process.DQMNone_step = cms.Path(process.DQMNone)
process.DQMMessageLoggerClientSeq_step = cms.Path(process.DQMMessageLoggerClientSeq)
process.DQMHarvestTrackerStrip_step = cms.Path(process.DQMHarvestTrackerStrip)
process.DQMCertTrackerStrip_step = cms.Path(process.DQMCertTrackerStrip)
process.DQMHarvestTrackerPixel_step = cms.Path(process.DQMHarvestTrackerPixel)
process.DQMCertTrackerPixel_step = cms.Path(process.DQMCertTrackerPixel)
process.DQMHarvestTracking_step = cms.Path(process.DQMHarvestTracking)
process.DQMCertTracking_step = cms.Path(process.DQMCertTracking)
process.DQMHarvestTrigger_step = cms.Path(process.DQMHarvestTrigger)
process.DQMCertTrigger_step = cms.Path(process.DQMCertTrigger)
process.DQMHarvestBeam_step = cms.Path(process.DQMHarvestBeam)
process.DQMHarvestFED_step = cms.Path(process.DQMHarvestFED)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.DQMHarvestMuon_step,process.DQMCertMuon_step,process.DQMHarvestL1T_step,process.DQMHarvestHcal_step,process.DQMHarvestJetMET_step,process.DQMCertJetMET_step,process.DQMHarvestEcal_step,process.DQMCertEcal_step,process.DQMHarvestEGamma_step,process.DQMCertEGamma_step,process.DQMNone_step,process.DQMMessageLoggerClientSeq_step,process.DQMHarvestTrackerStrip_step,process.DQMCertTrackerStrip_step,process.DQMHarvestTrackerPixel_step,process.DQMCertTrackerPixel_step,process.DQMHarvestTracking_step,process.DQMCertTracking_step,process.DQMHarvestTrigger_step,process.DQMCertTrigger_step,process.DQMHarvestBeam_step,process.DQMHarvestFED_step,process.dqmsave_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)



# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
