import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
from Configuration.StandardSequences.Simulation_cff import *
from SimGeneral.MixingModule.mixNoPU_cfi import *
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

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
myanalyzer = DQMEDAnalyzer('CaloTowersValidation',
    outputFile = cms.untracked.string('CaloTowersValidationHB.root'),
    CaloTowerCollectionLabel = cms.untracked.string('towerMaker'),
    hcalselector = cms.untracked.string('HB')
)

DQM.collectorHost = ''

#--- DigiToRaw <-> RawToDigi
from Configuration.StandardSequences.DigiToRaw_cff import *
from Configuration.StandardSequences.RawToDigi_cff  import *

### Special - CaloOnly ------------------------------------

#--- comes from DigiToRaw_cff.py
ecalPacker.Label = 'simEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"
#
#- hcalRawData (EventFilter/HcalRawToDigi/python/HcalDigiToRaw_cfi.py
#                 uses simHcalDigis by default...


#--- to force RAW->Digi
ecalDigis.InputLabel = 'rawDataCollector'
hcalDigis.InputLabel = 'rawDataCollector'
ecalPreshowerDigis.sourceTag = 'rawDataCollector'

#--- calolocalreco = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence)
# RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff
# RecoLocalCalo.Configuration.hcalLocalReco_cff


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

