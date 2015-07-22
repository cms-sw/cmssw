import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("hcalval")
process.load("Configuration.StandardSequences.Reconstruction_Data_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

#process.load("DQMServices.Core.DQM_cfg")
#process.DQM.collectorHost = ''

process.load("DQMServices.Core.DQMStore_cfi")
#process.load("DQMServices.Components.MEtoEDMConverter_cfi")

#process.load("DQMOffline.Configuration.DQMOffline_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2012D/JetHT/RECO/22Jan2013-v1/10004/1ACCAD9D-E496-E211-AA96-90E6BA19A25A.root',
        '/store/data/Run2012D/JetHT/RECO/22Jan2013-v1/10005/92C62727-2897-E211-8FDA-20CF3027A62F.root',
        '/store/data/Run2012D/JetHT/RECO/22Jan2013-v1/10015/DE3B5E5B-DE96-E211-BF85-002590747D92.root',
        '/store/data/Run2012D/JetHT/RECO/22Jan2013-v1/10015/F2E70090-DD96-E211-AA17-E0CB4EA0A8FE.root',
        '/store/data/Run2012D/JetHT/RECO/22Jan2013-v1/10015/FC59BBF3-DE96-E211-B868-E0CB4E1A11A1.root'
      ),
    inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')
)

#process.FEVT = cms.OutputModule("PoolOutputModule",
#     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
#     fileName = cms.untracked.string("HcalValHarvestingEDM.root")
#)

process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile               = cms.untracked.string('CaloTowersValidationRelVal.root'),
    CaloTowerCollectionLabel = cms.untracked.InputTag('towerMaker'),

    hcalselector             = cms.untracked.string('all'),
    mc                       = cms.untracked.string('no'),
    useAllHistos             = cms.untracked.bool(False)                         
)

process.noiseRates = cms.EDAnalyzer('NoiseRates',
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
#process.dqmSaver.saveByRun         = 1
#process.dqmSaver.saveAtJobEnd = False

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

##########
process.calotowersAnalyzer = cms.EDAnalyzer("CaloTowersAnalyzer",
     outputFile               = cms.untracked.string(''),
     CaloTowerCollectionLabel = cms.untracked.InputTag('towerMaker'),
     hcalselector             = cms.untracked.string('all'),
     useAllHistos             = cms.untracked.bool(False)
)
 
process.hcalRecHitsAnalyzer = cms.EDAnalyzer("HcalRecHitsAnalyzer",
#    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),
    outputFile                = cms.untracked.string(''),

    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),

    eventype                  = cms.untracked.string('multi'),
    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),
    useAllHistos              = cms.untracked.bool(False)                                 
)

process.hcalNoiseRates = cms.EDAnalyzer('HcalNoiseRates',
#    outputFile   = cms.untracked.string('NoiseRatesRelVal.root'),
    outputFile   = cms.untracked.string(''),
    rbxCollName  = cms.untracked.InputTag('hcalnoise'),
    minRBXEnergy = cms.untracked.double(20.0),
    minHitEnergy = cms.untracked.double(1.5),
    useAllHistos = cms.untracked.bool(False)
)

##########

process.p2 = cms.Path( process.hcalTowerAnalyzer * process.noiseRates * process.hcalRecoAnalyzer * 
                       process.calotowersAnalyzer * process.hcalRecHitsAnalyzer * process.hcalNoiseRates)
#                       * process.calotowersClient * process.noiseratesClient * process.hcalrechitsClient * process.dqmSaver)

#
# DQMIO
#
#process.DQMStore.enableMultiThread = cms.untracked.bool(False)
process.load('Configuration.EventContent.EventContent_cff')
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('file:DQMIO.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQMIO')
    )
)

#process.output = cms.EndPath(process.FEVT * process.DQMoutput)
process.output = cms.EndPath(process.DQMoutput)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
