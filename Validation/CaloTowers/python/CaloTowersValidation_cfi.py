import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
from Configuration.StandardSequences.Simulation_cff import *
from Configuration.StandardSequences.MixingNoPileUp_cff import *
from Configuration.StandardSequences.FakeConditions_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *
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

p = cms.Path(mix*calDigi*ecalLocalRecoSequence*hbhereco*hfreco*horeco*caloTowersRec*myanalyzer)
DQM.collectorHost = ''


