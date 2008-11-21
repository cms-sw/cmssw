import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "STARTUP_V5::All"
process.load("Validation.RecoMuon.PostProcessor_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:validationEDM.root")
)

process.DQMStore.referenceFileName = ""
process.DQMStore.collateHistograms = False

process.dqmSaver.convention = "Offline"
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings
process.dqmSaver.workflow = "/GlobalValidation/Test/RECO"

process.p1 = cms.Path(process.EDMtoMEConverter*
                      process.postProcessorMuonMultiTrack*process.postProcessorRecoMuon*
                      process.dqmSaver)
