import FWCore.ParameterSet.Config as cms

import FWCore.Utilities.FileUtils as FileUtils

process = cms.Process("validation")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("detailedInfo_RecoB_validation.txt")
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)

)



process.load("DQMServices.Core.DQM_cfg")

process.load("RecoBTag.Configuration.RecoBTag_cff")

process.load("PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi")  

process.load("Validation.RecoB.bTagAnalysis_cfi")
process.bTagValidation.jetMCSrc = 'AK5byValAlgo'
process.bTagValidation.allHistograms = True 
#process.bTagValidation.fastMC = True

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(10)
    input = cms.untracked.int32(-1)
)

#readFiles = cms.untracked.vstring( FileUtils.loadListFromFile ('bTag_PU25.txt') )

process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring("/store/user/cheung/phase1/363/R39/ttbar_btag_pu0/reco_1_1_MFX.root"),
     duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

#process.anal = cms.EDAnalyzer("EventContentAnalyzer")
#process.p7 = cms.Path(process.anal)

#process.plots = cms.Path(process.anal*process.myPartons* process.AK5Flavour * process.bTagValidation*process.dqmSaver)
process.plots = cms.Path(process.myPartons* process.AK5Flavour * process.bTagValidation*process.dqmSaver)
#process.plots = cms.Path(process.bTagValidation)
#process.plots = cms.Path(process.myPartons* process.AK5Flavour * process.impactParameterTagInfos* process.trackCountingHighEffBJetTags *process.bTagValidation)
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

#process.source.fileNames = cms.untracked.vstring("file:reco.root")
process.source.fileNames = cms.untracked.vstring( FileUtils.loadListFromFile ('bTag_fastPU50.txt') )
