import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("hcalval")
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# I added this hoping it will resolve hcaldigis
#process.load('Configuration/StandardSequences/DigiToRaw_cff')
#process.load('Configuration/StandardSequences/RawToDigi_cff')


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'DES19_62_V7::All', '')

#process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

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
#    fileNames = cms.untracked.vstring('file:QCD_30_35_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO.root'
    fileNames = cms.untracked.vstring(

#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/0C2E416F-E23E-E311-B27D-0026189438EF.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/160DD06F-E23E-E311-8248-00261894393C.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/1C5F274C-E33E-E311-B188-002618943979.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/20888BCF-F13E-E311-A2B6-002590593876.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/2E690227-E43E-E311-96E8-003048FFD7BE.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/3A7ACB32-E83E-E311-BCC0-003048FFD720.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/4A468041-E93E-E311-8FC3-00259059642A.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/4CA5FAF9-E63E-E311-88A9-0026189438B4.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/6E041CC3-E43E-E311-B90B-0026189438F7.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/720E7380-EF3E-E311-A359-0026189438C2.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/969D4B70-EC3E-E311-860D-003048FFD756.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/AA7203CE-ED3E-E311-88EF-0025905964A6.root',
#       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES17_62_V7_IdTest_UPG2017-v1/00000/B8237CE8-F03E-E311-8E71-0026189438E8.root'


       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/0679643C-E93E-E311-AC20-0025905964C2.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/0E1403CE-ED3E-E311-97C8-0025905964A6.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/1C56CB97-E33E-E311-8032-0025905964C2.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/2EC3AB8E-E73E-E311-BC69-00259059649C.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/48B92331-EF3E-E311-8DDC-002618943962.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/48E342A6-EA3E-E311-AE53-002590593878.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/64EF4EB7-E63E-E311-B13B-0025905938AA.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/6A5AF9C8-EB3E-E311-8AA2-00259059642A.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/7A0E3D6C-E63E-E311-B6BE-003048FFD76E.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/929955DE-E53E-E311-B198-003048FFD71A.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/B4CA3A70-E43E-E311-9678-0025905822B6.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/E4596378-E03E-E311-B4A7-0026189438D8.root',
       '/store/relval/CMSSW_6_2_0_SLHC2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_DES19_62_V7_IdTest_UPG2019-v1/00000/FA0E4D95-E13E-E311-955B-003048FFD760.root'
    ),
    #--- full set of GEN-SIM-DIGI-RAW(-HLTDEBUG) RelVal files -------------
    secondaryFileNames = cms.untracked.vstring(
     ),  
    inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')
)


process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     fileName = cms.untracked.string("HcalValHarvestingEDM.root")
)


process.hcalDigiAnalyzer = cms.EDAnalyzer("HcalDigisValidation",
    outputFile		      = cms.untracked.string('HcalDigisValidationRelVal.root'),
#    digiLabel                = cms.InputTag("simHcalDigis"),
#    subdetector               = cms.untracked.string('HE'),
    digiLabel   = cms.InputTag("hcalDigis"),  # regular collections
#--- Two Upgrade (doSLHC=True) collections
    digiLabelHBHE = cms.InputTag("simHcalDigis","HBHEUpgradeDigiCollection"), 
    digiLabelHF = cms.InputTag("simHcalDigis","HFUpgradeDigiCollection"),
    zside		      = cms.untracked.string('*'),
    mode		      = cms.untracked.string('multi'),
    hcalselector	      = cms.untracked.string('all'),
    mc			      = cms.untracked.string('yes'), # 'yes' for MC
    doSLHC                    = cms.untracked.bool(True) #  False for SLHC and True for regular rel val 
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


#--- NB: CHANGED for SLHC/Upgrade
process.hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),

### Upgrade 2019
    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbheUpgradeReco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfUpgradeReco"),
### 2017
#    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
#    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
#####
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),
    eventype                  = cms.untracked.string('multi'),
    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),
    mc                        = cms.untracked.string('no'),
    doSLHC                    = cms.untracked.bool(True) #  True for SLHC and False for regular rel val 
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


#process.hcalDigis.InputLabel = 'rawDataCollector' # MC
#---------------------------------------------------- Job PATH 
process.p2 = cms.Path( 
process.hcalTowerAnalyzer * 
process.hcalNoiseRates * 
process.hcalRecoAnalyzer *
process.hcalDigiAnalyzer * 
process.calotowersClient * 
process.noiseratesClient *
process.hcalrechitsClient * 
process.hcaldigisClient * 
process.dqmSaver)


#--- Customization for SLHC

from SLHCUpgradeSimulations.Configuration.HCalCustoms import customise_HcalPhase1
process=customise_HcalPhase1(process)

