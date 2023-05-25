import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('mtdValidation',Phase2C17I13M9)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')

process.load("Configuration.Geometry.GeometryExtended2026D95Reco_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')
process.load('RecoLocalFastTime.FTLClusterizer.MTDCPEESProducer_cfi')
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#Setup FWK for multithreaded
process.options.numberOfThreads = 4 # original
#process.options.numberOfThreads = 1
process.options.numberOfStreams = 0
process.options.numberOfConcurrentLuminosityBlocks = 0
process.options.eventSetup.numberOfConcurrentIOVs = 1

process.MessageLogger.cerr.FwkReport  = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10),
)

# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring(
#         'file:step3.root'
#     )
# )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'/store/relval/CMSSW_12_3_1/RelValZEE_14/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v10_2026D88PU200-v1/2580000/004d34e5-d815-439e-b5b5-7bf4197bd02b.root'
        '/store/relval/CMSSW_13_1_0_pre3/RelValZEE_14/GEN-SIM-RECO/PU_131X_mcRun4_realistic_v2_2026D95PU200-v1/00000/02e1c1df-11e7-4ba3-a827-3b5867a1d85a.root'
        # '/store/relval/CMSSW_12_3_1/RelValZEE_14/GEN-SIM-RECO/PU_123X_mcRun4_realistic_v10_2026D88PU200-v1/2580000/7206d4e3-1b30-44a0-8835-596834030974.root'
    )
)

process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)

# --- BTL Validation
process.load("Validation.MtdValidation.btlSimHitsValid_cfi")
process.load("Validation.MtdValidation.btlDigiHitsValid_cfi")
process.load("Validation.MtdValidation.btlLocalRecoValid_cfi")
btlValidation = cms.Sequence(process.btlSimHitsValid + process.btlDigiHitsValid + process.btlLocalRecoValid)

# --- ETL Validation
process.load("Validation.MtdValidation.etlSimHitsValid_cfi")
process.load("Validation.MtdValidation.etlDigiHitsValid_cfi")
process.load("Validation.MtdValidation.etlLocalRecoValid_cfi")
etlValidation = cms.Sequence(process.etlSimHitsValid + process.etlDigiHitsValid + process.etlLocalRecoValid)

# --- Global Validation
process.load("Validation.MtdValidation.mtdTracksValid_cfi")
process.load("Validation.MtdValidation.mtdEleIsoValid_cfi")
process.load("Validation.MtdValidation.vertices4DValid_cfi")

# process.btlDigiHitsValid.optionalPlots = True
# process.etlDigiHitsValid.optionalPlots = True
# process.btlLocalRecoValid.optionalPlots = True
# process.etlLocalRecoValid.optionalPlots = True
# process.mtdTracksValid.optionalPlots = True
# process.vertices4DValid.optionalPlots = True

process.validation = cms.Sequence(btlValidation + etlValidation + process.mtdTracksValid + process.mtdEleIsoValid + process.vertices4DValid)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.p = cms.Path( process.mix + process.mtdTrackingRecHits + process.validation )
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath( process.DQMoutput )

process.schedule = cms.Schedule( process.p , process.endjob_step , process.DQMoutput_step )
