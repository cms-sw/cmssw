import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("hcalval")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-RECO/MC_311_V1_64bit-v2/0012/E4D9AB0D-B72B-E011-98B3-0030487C2B86.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-RECO/MC_311_V1_64bit-v2/0011/F2D37EC6-282B-E011-95DC-001D09F2527B.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-RECO/MC_311_V1_64bit-v2/0011/D234573A-282B-E011-AA46-001D09F291D2.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-RECO/MC_311_V1_64bit-v2/0011/C2690CD4-272B-E011-AE30-0030487A18D8.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-RECO/MC_311_V1_64bit-v2/0011/B8987305-292B-E011-88C2-001D09F23A20.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-RECO/MC_311_V1_64bit-v2/0011/B44D0C89-292B-E011-AE7A-001D09F2AD4D.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-RECO/MC_311_V1_64bit-v2/0011/42EDCA5B-292B-E011-B5AB-001D09F24FBA.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-RECO/MC_311_V1_64bit-v2/0011/3E80EEB0-262B-E011-92ED-0030487A195C.root',
       '/store/relval/CMSSW_3_11_0/RelValTTbar/GEN-SIM-RECO/MC_311_V1_64bit-v2/0011/144AA7E6-292B-E011-A567-001D09F23A20.root'
      ),
    inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     fileName = cms.untracked.string("HcalValHarvestingEDM.root")
)

process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile               = cms.untracked.string('CaloTowersValidationRelVal.root'),
    CaloTowerCollectionLabel = cms.untracked.InputTag('towerMaker'),

    hcalselector             = cms.untracked.string('all'),
    mc                       = cms.untracked.string('no'),
    useAllHistos             = cms.untracked.bool(False)                         
)

process.hcalNoiseRates = cms.EDAnalyzer('NoiseRates',
    outputFile   = cms.untracked.string('NoiseRatesRelVal.root'),
    rbxCollName  = cms.untracked.InputTag('hcalnoise'),

    minRBXEnergy = cms.untracked.double(20.0),
    minHitEnergy = cms.untracked.double(1.5),
    useAllHistos = cms.untracked.bool(False)                         
)

process.hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),
    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),

    eventype                  = cms.untracked.string('multi'),
    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),
    mc                        = cms.untracked.string('no'),
    useAllHistos              = cms.untracked.bool(False)                                                                                                          
)

process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

process.calotowersClient = cms.EDAnalyzer("CaloTowersClient", 
     outputFile = cms.untracked.string('CaloTowersHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)

process.noiseratesClient = cms.EDAnalyzer("NoiseRatesClient", 
     outputFile = cms.untracked.string('NoiseRatesHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)

process.hcalrechitsClient = cms.EDAnalyzer("HcalRecHitsClient", 
     outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)

process.p2 = cms.Path( process.hcalTowerAnalyzer * process.hcalNoiseRates * process.hcalRecoAnalyzer
                       * process.calotowersClient * process.noiseratesClient * process.hcalrechitsClient * process.dqmSaver)
