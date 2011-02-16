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
process.load("Configuration.StandardSequences.Geometry_cff")
#process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#--- automatic GlobalTag setting -------------------------------------------
#--- to set it manually: - comment the following 2 lines
#--- and uncomment the 3d one with the actual tag to be set properly
from Configuration.PyReleaseValidation.autoCond import autoCond
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
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_64bit-v1/0090/48F08FAD-CC35-E011-9B28-001A92810ADE.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_64bit-v1/0089/C069192F-EF34-E011-B7F5-002618943962.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_64bit-v1/0089/B49B2DD3-FC34-E011-9FD8-001A92971BA0.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_64bit-v1/0089/AA249E1A-F834-E011-A138-001BFCDBD130.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_64bit-v1/0089/80C8298C-F934-E011-A744-0018F3C3E3A6.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_64bit-v1/0089/7C38E357-FE34-E011-9FE1-001A92971BD6.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_64bit-v1/0089/50965449-F634-E011-A84D-002354EF3BDF.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START311_V1_64bit-v1/0089/40EFDCE8-F834-E011-BE97-002618943933.root'
     ),
    #--- full set of GEN-SIM-DIGI-RAW(-HLTDEBUG) RelVal files -------------
    secondaryFileNames = cms.untracked.vstring(   
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0092/58791F53-EA35-E011-B9D2-002618943983.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/EE8CAAAA-FA34-E011-9157-002618943932.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/EA8462CA-FC34-E011-BC01-0018F3D096CA.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/D84F8B39-FC34-E011-9D8A-0026189438DD.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/CC4857CF-FD34-E011-A557-001A92811746.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/B41849E5-F734-E011-8A0B-002618943976.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/A81F43F1-F134-E011-BC11-0026189438B3.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/A6157263-F834-E011-A332-002618943860.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/A2CE34E7-F734-E011-8938-0018F3D09688.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/9C9EBB31-EE34-E011-9AC1-00248C55CC4D.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/88410887-F934-E011-BF30-0026189438BF.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/7E148456-FE34-E011-9F23-0026189437FA.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/60D671BE-F534-E011-97F3-001A92810AE6.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/38479DEA-F034-E011-B3FE-002618FDA216.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/368A48EC-F834-E011-B9FA-001BFCDBD11E.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/2A0714E7-F834-E011-A6A1-001A92971B06.root',
       '/store/relval/CMSSW_3_11_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START311_V1_64bit-v1/0089/248AB449-F634-E011-B791-00261894383E.root'
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
