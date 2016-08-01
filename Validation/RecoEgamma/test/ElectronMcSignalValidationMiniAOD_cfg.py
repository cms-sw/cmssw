
import sys
import os
import DQMOffline.EGamma.electronDataDiscovery as dd
import FWCore.ParameterSet.Config as cms

process = cms.Process("ElectronValidation")

process.options = cms.untracked.PSet( 
#    SkipEvent = cms.untracked.vstring('ProductNotFound'),
#    Rethrow = cms.untracked.vstring('ProductNotFound')
)

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

# OLD WAY

print "reading files ..."
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source.fileNames.extend(dd.search())
print "done"

# NEW WAY
#print "reading files ..."
#readFiles = cms.untracked.vstring()
#secFiles = cms.untracked.vstring() 
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
#process.source.fileNames.extend(dd.search())
#DD_SOURCE_TEMP = os.environ['DD_SOURCE']
#TEMP = os.environ['DD_SOURCE'].replace("MINIAODSIM", "GEN-SIM-RECO" ) 
#os.environ['DD_SOURCE'] = os.environ['DD_SOURCE'].replace("MINIAODSIM", "GEN-SIM-RECO" ) 
#process.source.secondaryFileNames.extend(dd.search2())
#os.environ['DD_SOURCE'] = DD_SOURCE_TEMP
#print "done"

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("FWCore.MessageService.MessageLogger_cfi")
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
#process.GlobalTag.globaltag = '75X_mcRun2_asymptotic_Queue'
#process.GlobalTag.globaltag = '75X_mcRun2_startup_Queue'

# FOR DATA REDONE FROM RAW, ONE MUST HIDE IsoFromDeps
# CONFIGURATION
process.load("Validation.RecoEgamma.electronIsoFromDeps_cff")
process.load("Validation.RecoEgamma.ElectronMcSignalValidatorMiniAOD_cfi")
process.load("Validation.RecoEgamma.ElectronIsolation_cfi")

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.EDM = cms.OutputModule("PoolOutputModule",
outputCommands = cms.untracked.vstring('drop *',"keep *_MEtoEDMConverter_*_*"),
fileName = cms.untracked.string(os.environ['TEST_HISTOS_FILE'].replace(".root", "_a.root"))
)

process.electronMcSignalValidatorMiniAOD.InputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD")
process.electronMcSignalValidatorMiniAOD.OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD")

process.p = cms.Path(process.ElectronIsolation * process.electronMcSignalValidatorMiniAOD * process.MEtoEDMConverter * process.dqmStoreStats)
#process.p = cms.Path(process.electronMcSignalValidatorMiniAOD * process.MEtoEDMConverter * process.dqmStoreStats)
#process.p = cms.Path(process.ElectronIsolation)

process.outpath = cms.EndPath(
process.EDM,
)
