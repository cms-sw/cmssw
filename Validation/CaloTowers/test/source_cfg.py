import FWCore.ParameterSet.Config as cms

process = cms.Process("RelValValidation")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['com10']

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#process.Timing = cms.Service("Timing")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RECO/May6thReReco-v1/0006/C6DD9B01-8459-DF11-BE72-003048678B16.root',
        '/store/data/Commissioning10/MinimumBias/RECO/May6thReReco-v1/0006/A42865DA-8659-DF11-B735-00304867920A.root',
        '/store/data/Commissioning10/MinimumBias/RECO/May6thReReco-v1/0006/38C42567-8259-DF11-AE1B-0018F3D09680.root'
      ),
    inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')
)

#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('132601:378-132601:381');

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     fileName = cms.untracked.string("HcalValHarvestingEDM.root")
)

process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile               = cms.untracked.string('CaloTowersValidationRelVal.root'),
    CaloTowerCollectionLabel = cms.untracked.InputTag('towerMaker'),
    hcalselector             = cms.untracked.string('all'),
    mc                       = cms.untracked.string('no')
)

process.hcalNoiseRates = cms.EDAnalyzer('NoiseRates',
    outputFile   = cms.untracked.string('NoiseRatesRelVal.root'),
    rbxCollName  = cms.untracked.InputTag('hcalnoise'),
    minRBXEnergy = cms.untracked.double(20.0),
    minHitEnergy = cms.untracked.double(1.5)
)

process.hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),

    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),

    eventype                  = cms.untracked.string('multi'),
    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),
    mc                        = cms.untracked.string('no')
)

process.p = cms.Path(process.hcalTowerAnalyzer * process.hcalNoiseRates * process.hcalRecoAnalyzer * process.MEtoEDMConverter)
process.output = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''
