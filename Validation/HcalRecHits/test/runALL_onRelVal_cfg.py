import FWCore.ParameterSet.Config as cms

process = cms.Process("RecHitsValidationRelVal")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/2008/4/7/RelVal-RelValQCD_Pt_50_80-1207529107/0000/0C938A10-7E04-DD11-B63B-000423D9890C.root')
)

process.hcalRecoAnalyzer = cms.EDFilter("HcalRecHitsValidation",
    eventype = cms.untracked.string('multi'),
    outputFile = cms.untracked.string('HcalRecHitsValidationALL_RelVal.root'),
    ecalselector = cms.untracked.string('yes'),
    mc = cms.untracked.string('no'),
    hcalselector = cms.untracked.string('all')
)

process.DQM.collectorHost = ''

process.p = cms.Path(process.hcalRecoAnalyzer)

