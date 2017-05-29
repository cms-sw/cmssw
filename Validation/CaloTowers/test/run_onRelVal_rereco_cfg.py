#################################################################
# Remaking HCAL RecHits and CaloTowers from RAW for validation  #
#################################################################

import os
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

process = cms.Process("RelValValidation")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#--- automatic GlobalTag setting -------------------------------------------
#--- to set it manually: - comment the following 2 lines
#--- and uncomment the 3d one with the actual tag to be set properly
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#######################################################################
#--- TWO-file approach, as both RAW  (for HCAL re-reco)    and
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

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     fileName = cms.untracked.string("HcalValHarvestingEDM.root")
)

process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile               = cms.untracked.string('CaloTowersValidationRelVal.root'),

    CaloTowerCollectionLabel = cms.untracked.InputTag('newtowerMaker'),
    hcalselector             = cms.untracked.string('all'),

    mc                       = cms.untracked.string('no'),
    useAllHistos             = cms.untracked.bool(False)                         
)

process.hcalNoiseRates = DQMEDHarvester('NoiseRates',
    outputFile   = cms.untracked.string('NoiseRatesRelVal.root'),

    rbxCollName  = cms.untracked.InputTag('newhcalnoise'),
    minRBXEnergy = cms.untracked.double(20.0),
    minHitEnergy = cms.untracked.double(1.5),

    useAllHistos = cms.untracked.bool(False)                         
)

process.hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),

    HBHERecHitCollectionLabel = cms.untracked.InputTag("newhbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("newhfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("newhoreco"),

    eventype                  = cms.untracked.string('multi'),
    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),

    mc                        = cms.untracked.string('no'),
)

#-----------------------------------------------------------------------------
#                        HCAL re-reco 4-step procedure preparation
#-----------------------------------------------------------------------------
#--- In case of DATA (re-reco) 
#--- one might need to add some parameters replacements
#--- similar to what is in the reference RECO config
#--- used for the original reconstruction !


#(1) -------------------------- to get hcalDigis (on the fly) from existing RAW
#
from EventFilter.HcalRawToDigi.HcalRawToDigi_cfi import *

#(2) -------------------------- to get (NEW) HCAL RecHits 
#
from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
process.newhbheprereco = hbheprereco.clone()
process.newhbhereco    = hbhereco.clone()
process.newhoreco      = horeco.clone()
process.newhfreco      = hfreco.clone()
process.newhbhereco.hbheInput    = "newhbheprereco"
process.newhcalLocalRecoSequence = cms.Sequence(process.newhbheprereco+process.newhbhereco+process.newhfreco+process.newhoreco)

#(3) -------------------------- to get (NEW) CaloTowers 
#
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import *
process.newtowerMaker = calotowermaker.clone()
process.newtowerMaker.hfInput = cms.InputTag("newhfreco")
process.newtowerMaker.hbheInput = cms.InputTag("newhbhereco")
process.newtowerMaker.hoInput = cms.InputTag("newhoreco")

#(4) -------------------------  to get (NEW) RBX noise 
# 
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
process.newhcalnoise = hcalnoise.clone()
process.newhcalnoise.recHitCollName = "newhbhereco"
process.newhcalnoise.caloTowerCollName = "newtowerMaker"

#Extra step: adding client post-processing
process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

process.calotowersClient = DQMEDHarvester("CaloTowersClient", 
     outputFile = cms.untracked.string('CaloTowersHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)

process.noiseratesClient = DQMEDHarvester("NoiseRatesClient", 
     outputFile = cms.untracked.string('NoiseRatesHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)

process.hcalrechitsClient = DQMEDHarvester("HcalRecHitsClient", 
     outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)


#--- Making re-reco and analysing
#--- first 4 producers: HCAL+CaloTowers(+RBX noise) re-reco. 
#
process.p = cms.Path(
process.hcalDigis *
process.newhcalLocalRecoSequence *
process.newtowerMaker *
process.newhcalnoise *
#--- analysis
process.hcalTowerAnalyzer * 
process.hcalNoiseRates * 
process.hcalRecoAnalyzer *
#--- post processing
process.calotowersClient *
process.noiseratesClient *
process.hcalrechitsClient *
process.dqmSaver
)
