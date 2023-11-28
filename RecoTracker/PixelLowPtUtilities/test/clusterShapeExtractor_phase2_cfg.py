import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2_cff import Phase2
process = cms.Process('RECO',Phase2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//02B67D97-9068-E711-BE9E-0025905B859E.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//0488FC9A-9468-E711-B4D7-0CC47A4C8E8A.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//064A9B36-9168-E711-8080-0025905A612C.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//0815DE39-9768-E711-9540-0025905B8568.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//0E9CA88B-9368-E711-9D2A-0025905B85D6.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//1A3A6461-8F68-E711-B864-002618FDA207.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//1C12F373-9B68-E711-82C8-0CC47A78A4BA.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//1C98A7EE-9068-E711-9B69-0CC47A7C34C8.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//20211387-9B68-E711-A8FA-0CC47A7C3424.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//341C3E77-8F68-E711-8D9B-0025905A60B8.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//34A50D6B-A268-E711-A4C7-0025905B8592.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//3C12037A-A168-E711-9257-0025905B8574.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//44DA0A1F-9868-E711-8515-0025905A609A.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//464212A2-9968-E711-9872-0CC47A4D75F0.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//4696F158-9168-E711-8A25-0CC47A4D76A0.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//48DDAE50-9368-E711-A5EC-0025905B8564.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//52EE641F-8F68-E711-A65B-0CC47A7C354C.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//5A7DD38D-9368-E711-AA19-0025905A4964.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//60AECE1A-9868-E711-B46E-0CC47A4C8E5E.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//70F4C291-9A68-E711-9F7B-0025905B858E.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//72FB1404-9668-E711-A65C-0025905B8612.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//74348000-9968-E711-A4F3-0025905B857A.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//A21FC2CB-9A68-E711-9CB4-0CC47A7C340C.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//BADBD400-9D68-E711-8D5A-0025905A612C.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//BE2BE4BC-9B68-E711-A160-0025905B8560.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//C8CD2AAB-9268-E711-ABF8-0025905A6066.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//D40F3BE9-9668-E711-8682-0025905A6122.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//D8CAE00E-9968-E711-AC88-0CC47A78A3EC.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//DA2969BC-8E68-E711-95A4-0CC47A4D765E.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//DAB487B8-9868-E711-A887-0025905B8568.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//E28E5E13-9968-E711-8B3D-0CC47A4D7600.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//E4681518-9D68-E711-B84A-0CC47A4D76B8.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//E8DD83DE-8F68-E711-8C16-0025905B860C.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//EA475C5F-9368-E711-9D6B-0025905B8594.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//EC560E76-9168-E711-80BC-0025905B85CC.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//EC934AFB-9768-E711-8542-003048FFD7CE.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//EEAFE2BB-8F68-E711-BDED-0025905A612C.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//F44C3502-9A68-E711-8E4E-0CC47A4D76CC.root',
    '/store/relval/CMSSW_9_3_0_pre2/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/92X_upgrade2023_realistic_v1_2023D17noPU-v1/00000//F6F7BA8D-9068-E711-BFC1-0CC47A7C345E.root',
),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True)
)


# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)

process.load('RecoTracker.PixelLowPtUtilities.ClusterShapeExtractor_cfi')
process.clusterShapeExtractor_step = cms.Path(process.clusterShapeExtractor)
process.clusterShapeExtractor.noBPIX1=False
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.clusterShapeExtractor_step)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(8)
process.options.numberOfStreams=cms.untracked.uint32(8)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions
process.load('Configuration.StandardSequences.PATMC_cff')

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC 

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
