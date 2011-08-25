#######################################################################
# Remaking HCAL Digis/RecHits and CaloTowers from RAW for validation  #
# version for >= CMSSW_390pre5
#######################################################################

import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("RelValValidation")
### process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
#process.load('Configuration.StandardSequences.RawToDigi_cff')
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

#-------------------------------- ANALYSERS
process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile               = cms.untracked.string('CaloTowersValidationRelVal.root'),
    CaloTowerCollectionLabel = cms.untracked.InputTag('newtowerMaker'),
    hcalselector             = cms.untracked.string('all'),
    mc                       = cms.untracked.string('no'),
    useAllHistos             = cms.untracked.bool(False)  
)

process.hcalNoiseRates = cms.EDAnalyzer('NoiseRates',
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
    useAllHistos              = cms.untracked.bool(False)                                                                                                          
)

#-----------------------------------------------------------------------------
#                     HCAL re-digi preparation
#                     requires explicit random definition
#-----------------------------------------------------------------------------
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    newsimHcalUnsuppressedDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(11223344),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)
process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import *
from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *

process.newsimHcalUnsuppressedDigis     = simHcalUnsuppressedDigis.clone()
process.newsimHcalDigis                 = simHcalDigis.clone()
process.newsimHcalDigis.digiLabel       = "newsimHcalUnsuppressedDigis"

process.newhcalDigiSequence = cms.Sequence(process.newsimHcalUnsuppressedDigis+process.newsimHcalDigis)


#-----------------------------------------------------------------------------
#                     HCAL re-reco 3-step procedure preparation
#-----------------------------------------------------------------------------
#--- In case of DATA (re-reco) 
#--- one might need to add some parameters replacements
#--- similar to what is in the reference RECO config
#--- used for the original reconstruction !

#(1) -------------------------- to get (NEW) HCAL RecHits 
#
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
process.newhbheprereco = hbheprereco.clone()
process.newhoreco      = horeco.clone()
process.newhfreco      = hfreco.clone()
process.newhbhereco    = hbhereco.clone() 
process.newhbheprereco.digiLabel = "newsimHcalDigis"    
process.newhoreco.digiLabel      = "newsimHcalDigis"  
process.newhfreco.digiLabel      = "newsimHcalDigis"
process.newhbhereco.hbheInput    = "newhbheprereco"

process.newhcalLocalRecoSequence = cms.Sequence(process.newhbheprereco+process.newhbhereco+process.newhfreco+process.newhoreco)

#(2) -------------------------- to get (NEW) CaloTowers 
#
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import *
process.newtowerMaker = calotowermaker.clone()
process.newtowerMaker.hfInput = cms.InputTag("newhfreco")
process.newtowerMaker.hbheInput = cms.InputTag("newhbhereco")
process.newtowerMaker.hoInput = cms.InputTag("newhoreco")

#(3) -------------------------  to get (NEW) RBX noise 
# 
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
process.newhcalnoise = hcalnoise.clone()
process.newhcalnoise.digiCollName      = "newsimHcalDigis"
process.newhcalnoise.recHitCollName    = "newhbhereco"
process.newhcalnoise.caloTowerCollName = "newtowerMaker"

#-----------------------------------------------------------------------------
#                     Extra step: adding client post-processing
#-----------------------------------------------------------------------------
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

#--- Mixing is required when reading PCaloHits for re-Digitization
from SimGeneral.MixingModule.mixNoPU_cfi import *

#--------------------------- Making re-digi/re-reco and analysing
#
process.p = cms.Path(
process.mix *
process.newhcalDigiSequence *
process.newhcalLocalRecoSequence *
process.newtowerMaker *
process.newhcalnoise *
#--- analysis
process.hcalTowerAnalyzer * 
process.hcalNoiseRates * 
process.hcalRecoAnalyzer *
#--- post-processing
process.calotowersClient *
process.noiseratesClient *
process.hcalrechitsClient *
process.dqmSaver
)
