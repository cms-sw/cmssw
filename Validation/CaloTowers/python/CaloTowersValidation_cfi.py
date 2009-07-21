import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
from Configuration.StandardSequences.Simulation_cff import *
from Configuration.StandardSequences.MixingNoPileUp_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
GlobalTag.globaltag = 'MC_31X_V3::All'

from DQMServices.Core.DQM_cfg import *
maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:/')
)

MessageLogger = cms.Service("MessageLogger")

myanalyzer = cms.EDFilter("CaloTowersValidation",
    outputFile = cms.untracked.string('CaloTowersValidationHB.root'),
    CaloTowerCollectionLabel = cms.untracked.string('towerMaker'),
    hcalselector = cms.untracked.string('HB')
)

DQM.collectorHost = ''

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


p = cms.Path(
 mix *
 calDigi *
 ecalPacker *
 esDigiToRaw *
 hcalRawData *
 rawDataCollector *
 ecalDigis *
 ecalPreshowerDigis *
 hcalDigis *
 calolocalreco *
 caloTowersRec *
 myanalyzer
)

