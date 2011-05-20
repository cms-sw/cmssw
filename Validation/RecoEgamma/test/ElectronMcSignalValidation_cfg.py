
import sys
import os
import DQMOffline.EGamma.electronDbsDiscovery as dbs
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronValidation")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source.fileNames.extend(dbs.search())

process.load("Validation.RecoEgamma.ElectronMcSignalValidator_cfi")

process.electronMcSignalValidator.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
#process.electronMcSignalValidator.OutputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")

process.p = cms.Path(process.electronMcSignalValidator*process.dqmStoreStats)


