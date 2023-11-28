# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:phase1_2017_realistic -n 10 --era Run2_2017 --eventcontent RECOSIM,DQM --runUnscheduled -s RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM --datatier GEN-SIM-RECO,DQMIO --geometry DB:Extended --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process('RECO',Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//04A76B1B-CF60-E711-BB55-0CC47A4C8EC8.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//120DA883-CF60-E711-B945-0025905A60B6.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//2076661C-CF60-E711-9A0D-0025905A60B8.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//2C06018A-D060-E711-88F4-0025905B85D8.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//34F2166D-CF60-E711-876B-0025905B85DE.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//3E937FAE-CF60-E711-A74C-0025905B855A.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//486BB49C-CF60-E711-9205-0CC47A7C35C8.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//522B4781-CF60-E711-8BF5-0025905B8560.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//56D8A4B8-CF60-E711-BDF8-003048FFCC16.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//86A1DCDC-CF60-E711-B676-0CC47A7C35D8.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//8ACD778E-D060-E711-86C1-0CC47A4D7606.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//98F51D6D-CF60-E711-9471-0CC47A7C34EE.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//A8D802D6-CF60-E711-A4DF-0CC47A78A41C.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//BCA2973A-DA60-E711-949B-0025905A611C.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//D88729CC-CE60-E711-913C-0CC47A4C8E3C.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//DC3649D5-CF60-E711-93FB-0CC47A745250.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//E011547C-D060-E711-9DDD-0CC47A4C8E3C.root',
'/store/relval/CMSSW_9_3_0_pre1/RelValMinBias_13/GEN-SIM-DIGI-RAW/92X_upgrade2017_realistic_v7-v1/00000//F40A90CF-CF60-E711-8F1A-0CC47A7C35A8.root',
),
    secondaryFileNames = cms.untracked.vstring()
)


#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(True)
)

process.Timing = cms.Service("Timing"
    ,summaryOnly = cms.untracked.bool(True)
)


process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
    numberOfThreads = cms.untracked.uint32(8),
    numberOfStreams = cms.untracked.uint32(8),
    wantSummary = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:step3.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
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

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction_trackingOnly)
process.prevalidation_step = cms.Path(process.globalPrevalidationTrackingOnly)
process.load('RecoTracker.PixelLowPtUtilities.ClusterShapeExtractor_cfi')
process.clusterShapeExtractor.noBPIX1=False
process.clusterShapeExtractor_step = cms.Path(process.clusterShapeExtractor)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.clusterShapeExtractor_step)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
