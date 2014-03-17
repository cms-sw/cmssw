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
process.GlobalTag = GlobalTag(process.GlobalTag, 'DES19_62_V8::All', '')

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
#       '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/24E76D2D-0FA1-E311-B745-02163E00EA2B.root',
       '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/4E183A9F-EEA0-E311-A057-0019B9F3F4C8.root',
       '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/723D1E95-07A1-E311-8314-0025B3203886.root',
       '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/8852B1D4-09A1-E311-8BC8-003048673036.root',
       '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/90713810-F3A0-E311-BCF8-02163E00E62E.root',
       '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/92DA4D1A-04A1-E311-ABBA-003048946F74.root',
       '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/9AB945AB-FDA0-E311-8F05-0025B320383C.root',
       '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/9C95C703-FAA0-E311-952B-0025904B11AE.root',
       '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/BC661835-22A1-E311-B271-02163E00E6CD.root'


##       '/store/relval/CMSSW_6_2_0_SLHC5/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/22164515-0B69-E311-9380-003048FFCC18.root',
#       '/store/relval/CMSSW_6_2_0_SLHC5/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/2E4B8548-1F69-E311-9CAE-0025905A60F2.root',
##       '/store/relval/CMSSW_6_2_0_SLHC5/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/6028CD2D-0B69-E311-94F4-0025905A6068.root',
##       '/store/relval/CMSSW_6_2_0_SLHC5/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/6C73969E-0B69-E311-A07F-0025905A610C.root',
##       '/store/relval/CMSSW_6_2_0_SLHC5/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/7454BEC2-2669-E311-8067-002618943821.root',
#       '/store/relval/CMSSW_6_2_0_SLHC5/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/74A00327-1D69-E311-A8E7-0025905A497A.root',
##       '/store/relval/CMSSW_6_2_0_SLHC5/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/84E5C695-1869-E311-95C2-0025905A6068.root',
#       '/store/relval/CMSSW_6_2_0_SLHC5/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/E64CE02A-3269-E311-BE61-0026189438C0.root'
##       '/store/relval/CMSSW_6_2_0_SLHC5/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v1/00000/F614ABE2-1369-E311-8878-0025905A60D0.root'


#       '/store/relval/CMSSW_6_2_0_SLHC7/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v2/00000/3007C1E4-9390-E311-A43B-002590494C20.root',
#       '/store/relval/CMSSW_6_2_0_SLHC7/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v2/00000/46AB651E-5B90-E311-8A87-02163E00E705.root',
#       '/store/relval/CMSSW_6_2_0_SLHC7/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v2/00000/64F20459-4E90-E311-8A01-02163E00E6ED.root',
#       '/store/relval/CMSSW_6_2_0_SLHC7/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v2/00000/700F7B57-5190-E311-A836-02163E00EA91.root',
#       '/store/relval/CMSSW_6_2_0_SLHC7/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v2/00000/726D4F5C-6890-E311-BC0D-02163E00E864.root',
#       '/store/relval/CMSSW_6_2_0_SLHC7/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v2/00000/9267FE95-6A90-E311-A3E5-02163E00EB64.root',
#       '/store/relval/CMSSW_6_2_0_SLHC7/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v2/00000/B2882E67-5590-E311-948C-02163E00EB5D.root',
#       '/store/relval/CMSSW_6_2_0_SLHC7/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v2/00000/D68624BC-5B90-E311-A234-02163E00E5E2.root',
#       '/store/relval/CMSSW_6_2_0_SLHC7/RelValTTbar_14TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v2/00000/E6EA308E-5490-E311-BC48-0025B32440A4.root'
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
    digiLabel                 = cms.InputTag("simHcalDigis"),
    digiLabelHBHE             = cms.InputTag("simHcalDigis","HBHEUpgradeDigiCollection"),
    digiLabelHF               = cms.InputTag("simHcalDigis","HFUpgradeDigiCollection"),
#    subdetector               = cms.untracked.string('HE'),

#### for 2017 data sets
#    HBHEDigisCollectionLabel = cms.InputTag("simHcalDigis"),
#    HFDigisCollectionLabel   = cms.InputTag("simHcalDigis"),
#    HODigisCollectionLabel   = cms.InputTag("simHcalDigis"),
#### for 2019 data sets
#    HBHEDigisCollectionLabel = cms.InputTag("simHcalDigis","HBHEUpgradeDigiCollection"),
#    HFDigisCollectionLabel   = cms.InputTag("simHcalDigis","HFUpgradeDigiCollection"),
#    HODigisCollectionLabel   = cms.InputTag("simHcalDigis"),

    zside		      = cms.untracked.string('*'),
    mode		      = cms.untracked.string('multi'),
    hcalselector	      = cms.untracked.string('all'),
    mc			      = cms.untracked.string('yes'), # 'yes' for MC
    doSLHC                    = cms.untracked.bool(True) #  True for SLHC and False for regular rel val 
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

