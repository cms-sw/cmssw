import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("hcalval")
process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )

#######################################################################
# TWO-file approach, as both RAW  (for HCAL re-reco)    and
#                               RECO (for unchanged ECAL)  are required 
#######################################################################
process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),   
    #--- full set of GEN-SIM-RECO RelVal files ----------------------------
    fileNames = cms.untracked.vstring(

    ),
    #--- full set of GEN-SIM-DIGI-RAW(-HLTDEBUG) RelVal files -------------
    secondaryFileNames = cms.untracked.vstring(   

     ),  
    inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')
)


process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     fileName = cms.untracked.string("HcalValHarvestingEDM.root")
)


process.hcalDigiAnalyzer = cms.EDAnalyzer("HcalDigisValidation",
    outputFile		      = cms.untracked.string('HcalDigisValidationRelVal.root'),
    digiLabel		      = cms.InputTag("hcalDigis"),
    zside		      = cms.untracked.string('*'),
    mode		      = cms.untracked.string('multi'),

    hcalselector	      = cms.untracked.string('all'),
    mc			      = cms.untracked.string('yes') # 'yes' for MC
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

process.hcaldigisClient = cms.EDAnalyzer("HcalDigisClient",
     outputFile	= cms.untracked.string('HcalDigisHarvestingME.root'),
     DQMDirName	= cms.string("/") # root directory
)   

process.hcalDigis.InputLabel = 'rawDataCollector' # MC

process.p2 = cms.Path( 
process.hcalDigis * 
process.hcalTowerAnalyzer * 
process.hcalNoiseRates * 
process.hcalRecoAnalyzer *
process.hcalDigiAnalyzer * 
process.calotowersClient * 
process.noiseratesClient *
process.hcalrechitsClient * 
process.hcaldigisClient * 
process.dqmSaver)
