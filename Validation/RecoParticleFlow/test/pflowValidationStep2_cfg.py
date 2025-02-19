# test file for PFDQM Harvesting
# the eEDM file(s) created in first step with DQM histograms are hasvested
# (summed up & extracted) and a root file with histograms are created
# The PFlow DQM client is used here to perform some analysis (fit and
# extraction of mean, sigma etc.) on histograms after full statistics is available
process = cms.Process('PFlowHarvest')
#------------------------
# Message Logger Settings
#------------------------
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 10000
#------------------------------------------------
# Input Source (EDM file(s) created at step #1
#-----------------------------------------------
process.source = cms.Source("PoolSource",
                   fileNames = cms.untracked.vstring('file:MEtoEDM_PFlow.root'),
                   processingMode = cms.untracked.string('RunsAndLumis')
                 )
#-----------------------
# EDM to ME (histogram) 
#-----------------------
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
#--------------------------------------------
# Core DQM and definition of output EDM file
#--------------------------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.producer = 'DQM'
process.dqmSaver.convention = 'Offline'
# specify part of your output file name
process.dqmSaver.workflow = '/PFlow/Validation/QCD'
process.dqmEnv.subSystemFolder = 'ParticleFlow'
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 1

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('NOMERGE')
)
#--------------
# PFlow Client 
#--------------
process.load("Validation.RecoParticleFlow.PFValidationClient_cff")

process.pfClientSequence = cms.Sequence(process.pfJetClient*process.pfMETClient*process.pfElectronClient)
process.p = cms.Path(process.EDMtoME*process.pfClientSequence*process.dqmEnv*process.dqmSaver)

