import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
from Configuration.StandardSequences.VtxSmearedGauss_cff import *
from Configuration.StandardSequences.Simulation_cff import *
from Configuration.StandardSequences.MixingNoPileUp_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
GlobalTag.globaltag = 'MC_31X_V3::All'

from DQMServices.Core.DQM_cfg import *
MessageLogger = cms.Service("MessageLogger")


source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/1_4_x/mc_pi+100_etaphi44.root')
)

hcalDigiAnalyzer = cms.EDFilter("HcalDigiTester",
    digiLabel = cms.InputTag("simHcalDigis"),
    subpedvalue = cms.untracked.bool(True),
    outputFile = cms.untracked.string('HcalDigisValidationHF.root'),
    hcalselector = cms.untracked.string('HF')
)

hcalRecoAnalyzer = cms.EDFilter("HcalRecHitsValidation",
    outputFile = cms.untracked.string('HcalRecHitsValidationHB.root'),
    eventype = cms.untracked.string('single'),
    mc = cms.untracked.string('yes'),
    sign = cms.untracked.string('*'),
    hcalselector = cms.untracked.string('HF'),
    ecalselector = cms.untracked.string('no')
)

hcalTowerAnalyzer = cms.EDFilter("CaloTowersValidation",
    outputFile = cms.untracked.string('CaloTowersValidationHB.root'),
    CaloTowerCollectionLabel = cms.untracked.string('towerMaker'),
    hcalselector = cms.untracked.string('HB')
)

DQM.collectorHost = ''

VtxSmeared.SigmaX = 0.00001
VtxSmeared.SigmaY = 0.00001
VtxSmeared.SigmaZ = 0.00001


#--- DigiToRaw <-> RawToDigi
from Configuration.StandardSequences.DigiToRaw_cff import *
from Configuration.StandardSequences.RawToDigi_cff  import *


### Special - CaloOnly ---------------------------------------------------
ecalGlobalUncalibRecHit.EBdigiCollection = cms.InputTag("ecalDigis","ebDigis")
ecalGlobalUncalibRecHit.EEdigiCollection = cms.InputTag("ecalDigis","eeDigis")
ecalPreshowerRecHit.ESdigiCollection = cms.InputTag("ecalPreshowerDigis") 
hbhereco.digiLabel = cms.InputTag("hcalDigis")
horeco.digiLabel   = cms.InputTag("hcalDigis")
hfreco.digiLabel   = cms.InputTag("hcalDigis")
ecalRecHit.recoverEBIsolatedChannels = cms.bool(False)
ecalRecHit.recoverEEIsolatedChannels = cms.bool(False)
ecalRecHit.recoverEBFE = cms.bool(False)
ecalRecHit.recoverEEFE = cms.bool(False)





