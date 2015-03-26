import FWCore.ParameterSet.Config as cms

process = cms.Process("RecHitsValidationRelVal")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")


process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
      )
)

process.hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
    eventype = cms.untracked.string('multi'),
    outputFile = cms.untracked.string('HcalRecHitsValidationALL_RelVal.root'),
    ecalselector = cms.untracked.string('yes'),
    mc = cms.untracked.string('no'),
    hcalselector = cms.untracked.string('all')
)


process.p = cms.Path(process.hcalRecoAnalyzer)

