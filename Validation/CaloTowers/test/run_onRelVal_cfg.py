import FWCore.ParameterSet.Config as cms

process = cms.Process("RelValValidation")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")


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

process.hcalRecoAnalyzer = cms.EDFilter("HcalRecHitsValidation",
    eventype = cms.untracked.string('multi'),
    outputFile = cms.untracked.string('HcalRecHitValidationRelVal.root'),
    ecalselector = cms.untracked.string('yes'),
    mc = cms.untracked.string('no'),
    hcalselector = cms.untracked.string('all')
)


process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile = cms.untracked.string('CaloTowersValidationRelVal.root'),
    CaloTowerCollectionLabel = cms.untracked.string('towerMaker'),
    hcalselector = cms.untracked.string('all')
)

process.hcalNoiseRates = cms.EDAnalyzer('NoiseRates',
     rbxCollName = cms.string('hcalnoise'),
     outputFile = cms.untracked.string('NoiseRatesRelVal.root'),
     minRBXEnergy = cms.double(20.0),
     minHitEnergy = cms.double(1.5)
)

process.p = cms.Path(process.hcalRecoAnalyzer * process.hcalTowerAnalyzer * process.hcalNoiseRates)

