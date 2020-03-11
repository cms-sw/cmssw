# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step4 --mc --filetype DQM --conditions auto:run2_mc_FULL -s HARVESTING:@allForPrompt --era Run2_2018 --scenario pp --filein file:RECO_RAW2DIGI_L1Reco_RECO_EI_PAT_DQM_inDQM.root --python_filename=harvesting.py --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

process = cms.Process('HARVESTING',Run2_2018)

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
    fileNames = cms.untracked.vstring('file:RECO_RAW2DIGI_L1Reco_RECO_EI_PAT_DQM_inDQM.root')
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    SkipEvent = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(

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
    annotation = cms.untracked.string('step4 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_FULL', '')

# Path and EndPath definitions
process.genHarvesting = cms.Path(process.postValidation_gen)
process.dqmHarvestingFakeHLT = cms.Path(process.DQMOffline_SecondStep_FakeHLT+process.DQMOffline_Certification)
process.validationHarvestingHI = cms.Path(process.postValidationHI)
process.dqmHarvestingExtraHLT = cms.Path(process.DQMOffline_SecondStep_ExtraHLT+process.DQMOffline_Certification)
process.alcaHarvesting = cms.Path()
process.validationHarvestingNoHLT = cms.Path(process.postValidation+process.postValidation_gen)
process.validationHarvestingFS = cms.Path(process.recoMuonPostProcessors+process.postValidationTracking+process.MuIsoValPostProcessor+process.calotowersPostProcessor+process.hcalSimHitsPostProcessor+process.hcaldigisPostProcessor+process.hcalrechitsPostProcessor+process.electronPostValidationSequence+process.photonPostProcessor+process.pfJetClient+process.pfMETClient+process.pfJetResClient+process.pfElectronClient+process.rpcRecHitPostValidation_step+process.makeBetterPlots+process.bTagCollectorSequenceMCbcl+process.METPostProcessor+process.L1GenPostProcessor+process.bdHadronTrackPostProcessor+process.siPixelPhase1OfflineDQM_harvestingV+process.MuonGEMHitsPostProcessors+process.MuonGEMDigisPostProcessors+process.MuonGEMRecHitsPostProcessors+process.postValidation_gen)
process.validationpreprodHarvesting = cms.Path(process.postValidation_preprod+process.hltpostvalidation_preprod+process.postValidation_gen)
process.validationprodHarvesting = cms.Path(process.hltpostvalidation_prod+process.postValidation_gen)
process.validationHarvesting = cms.Path(process.postValidation+process.hltpostvalidation+process.postValidation_gen)
process.validationpreprodHarvestingNoHLT = cms.Path(process.postValidation_preprod+process.postValidation_gen)
process.dqmHarvestingPOGMC = cms.Path(process.DQMOffline_SecondStep_PrePOGMC)
process.validationHarvestingMiniAOD = cms.Path(process.JetPostProcessor+process.METPostProcessorHarvesting+process.postValidationMiniAOD)
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep+process.DQMOffline_Certification)
process.DQMHarvestMuon_step = cms.Path(process.DQMHarvestMuon)
process.DQMCertMuon_step = cms.Path(process.DQMCertMuon)
process.DQMHarvestHcal_step = cms.Path(process.DQMHarvestHcal)
process.DQMHarvestJetMET_step = cms.Path(process.DQMHarvestJetMET)
process.DQMCertJetMET_step = cms.Path(process.DQMCertJetMET)
process.DQMHarvestEcal_step = cms.Path(process.DQMHarvestEcal)
process.DQMCertEcal_step = cms.Path(process.DQMCertEcal)
process.DQMHarvestEGamma_step = cms.Path(process.DQMHarvestEGamma)
process.DQMCertEGamma_step = cms.Path(process.DQMCertEGamma)
process.DQMHarvestDCS_step = cms.Path(process.DQMHarvestDCS)
process.DQMMessageLoggerClientSeq_step = cms.Path(process.DQMMessageLoggerClientSeq)
process.DQMHarvestTrackerStrip_step = cms.Path(process.DQMHarvestTrackerStrip)
process.DQMCertTrackerStrip_step = cms.Path(process.DQMCertTrackerStrip)
process.DQMHarvestTrackerPixel_step = cms.Path(process.DQMHarvestTrackerPixel)
process.DQMCertTrackerPixel_step = cms.Path(process.DQMCertTrackerPixel)
process.DQMHarvestTracking_step = cms.Path(process.DQMHarvestTracking)
process.DQMCertTracking_step = cms.Path(process.DQMCertTracking)
process.DQMHarvestL1T_step = cms.Path(process.DQMHarvestL1T)
process.DQMHarvestTrigger_step = cms.Path(process.DQMHarvestTrigger)
process.DQMCertTrigger_step = cms.Path(process.DQMCertTrigger)
process.DQMHarvestBeam_step = cms.Path(process.DQMHarvestBeam)
process.DQMHarvestFED_step = cms.Path(process.DQMHarvestFED)
process.DQMHarvestTAU_step = cms.Path(process.DQMHarvestTAU)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.DQMHarvestMuon_step,process.DQMCertMuon_step,process.DQMHarvestHcal_step,process.DQMHarvestJetMET_step,process.DQMCertJetMET_step,process.DQMHarvestEcal_step,process.DQMCertEcal_step,process.DQMHarvestEGamma_step,process.DQMCertEGamma_step,process.DQMHarvestDCS_step,process.DQMMessageLoggerClientSeq_step,process.DQMHarvestTrackerStrip_step,process.DQMCertTrackerStrip_step,process.DQMHarvestTrackerPixel_step,process.DQMCertTrackerPixel_step,process.DQMHarvestTracking_step,process.DQMCertTracking_step,process.DQMHarvestL1T_step,process.DQMHarvestTrigger_step,process.DQMCertTrigger_step,process.DQMHarvestBeam_step,process.DQMHarvestFED_step,process.DQMHarvestTAU_step,process.dqmsave_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
