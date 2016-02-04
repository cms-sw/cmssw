
import sys
import os
import electronDbsDiscovery
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronValidation")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source.fileNames.extend(electronDbsDiscovery.search())

process.load("Validation.RecoEgamma.ElectronMcFakeValidator_cfi")

process.electronMcFakeValidator.outputFile = cms.string(os.environ['TEST_OUTPUT_FILE'])
#process.electronMcFakeValidator.histosCfg = cms.PSet(process.electronMcFakeFineBining)

process.p = cms.Path(process.electronMcFakeValidator*process.dqmStoreStats)


