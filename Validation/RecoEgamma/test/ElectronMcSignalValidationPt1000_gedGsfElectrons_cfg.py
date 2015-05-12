
import sys
import os
import DQMOffline.EGamma.electronDataDiscovery as dd
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronValidation")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source.fileNames.extend(dd.search())

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff") # new 
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']#+'::All'

# FOR DATA REDONE FROM RAW, ONE MUST HIDE IsoFromDeps
# CONFIGURATION
process.load("Validation.RecoEgamma.electronIsoFromDeps_cff")
process.load("Validation.RecoEgamma.ElectronMcSignalValidatorPt1000_gedGsfElectrons_cfi")

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.EDM = cms.OutputModule("PoolOutputModule",
outputCommands = cms.untracked.vstring('drop *',"keep *_MEtoEDMConverter_*_*"),
fileName = cms.untracked.string(os.environ['TEST_HISTOS_FILE'].replace(".root", "_a.root"))
)

process.electronMcSignalValidatorPt1000.InputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorPt1000")
process.electronMcSignalValidatorPt1000.OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorPt1000")

process.p = cms.Path(process.electronMcSignalValidatorPt1000 * process.MEtoEDMConverter * process.dqmStoreStats)

process.outpath = cms.EndPath(
process.EDM,
)
