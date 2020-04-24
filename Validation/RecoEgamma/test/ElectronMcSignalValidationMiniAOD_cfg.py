
import sys
import os
import DQMOffline.EGamma.electronDataDiscovery as dd
import FWCore.ParameterSet.Config as cms

process = cms.Process("ElectronValidation")

process.options = cms.untracked.PSet( 
#    SkipEvent = cms.untracked.vstring('ProductNotFound'),
#    Rethrow = cms.untracked.vstring('ProductNotFound')
)

#from FWCore.Modules.printContent_cfi import * 

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

# OLD WAY

print "reading files ..."
max_number = -1 # number of events
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(max_number))
process.source = cms.Source ("PoolSource",
#eventsToProcess = cms.untracked.VEventRange('1:2682-1:2682'), 
#eventsToProcess = cms.untracked.VEventRange('1:8259-1:8259'), 
#eventsToProcess = cms.untracked.VEventRange('1:10-1:10'), 
fileNames = cms.untracked.vstring([
#        'file:PAT_249120E2-D1EC-E611-83C2-0CC47A7C347A.root',
#        'file:PAT_76F9AD07-D3EC-E611-AA87-0CC47A745250.root',
#        'file:PAT_FA0E1D02-D5EC-E611-B8CA-0025905A6080.root',
#        'file:PAT_EE728E01-D5EC-E611-9DC5-0025905A6126.root',
#        '/store/relval/CMSSW_9_1_0_pre2/RelValSingleElectronPt10/MINIAODSIM/90X_upgrade2017_realistic_v20-v1/00000/66B5E60E-5619-E711-8E17-0CC47A4D7632.root',
#        '/store/relval/CMSSW_9_1_0_pre2/RelValSingleElectronPt10/MINIAODSIM/90X_upgrade2017_realistic_v20-v1/00000/BC08FC0E-5619-E711-98E5-0CC47A4D760C.root',
    ]),
secondaryFileNames = cms.untracked.vstring() )
process.source.fileNames.extend(dd.search())
print "reading files done"

#process.printTree = cms.EDAnalyzer("ParticleListDrawer",
#  maxEventsToPrint = cms.untracked.int32(1),
#  printVertex = cms.untracked.bool(False),
#  printOnlyHardInteraction = cms.untracked.bool(False), # Print only status=3 particles. This will not work for Pythia8, which does not have any such particles.
#  src = cms.InputTag("prunedGenParticles")
#)

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
#process.GlobalTag.globaltag = '90X_upgrade2017_realistic_v20'
process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']#+'::All'
#process.GlobalTag.globaltag = '75X_mcRun2_asymptotic_Queue'
#process.GlobalTag.globaltag = '75X_mcRun2_startup_Queue'

# FOR DATA REDONE FROM RAW, ONE MUST HIDE IsoFromDeps
# CONFIGURATION
process.load("Validation.RecoEgamma.electronIsoFromDeps_cff")
process.load("Validation.RecoEgamma.ElectronIsolation_cfi")
process.load("Validation.RecoEgamma.ElectronMcSignalValidatorMiniAOD_cfi")

print "miniAODElectronIsolation call"
from Validation.RecoEgamma.electronValidationSequenceMiniAOD_cff import miniAODElectronIsolation # as _ElectronIsolationCITK
process.miniAODElectronIsolation = miniAODElectronIsolation
print "miniAODElectronIsolation clone done"

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#process.printContent = printContent

process.EDM = cms.OutputModule("PoolOutputModule",
outputCommands = cms.untracked.vstring('drop *',"keep *_MEtoEDMConverter_*_*"),
#fileName = cms.untracked.string(TEST_HISTOS_FILE)
fileName = cms.untracked.string(os.environ['TEST_HISTOS_FILE'].replace(".root", "_a.root"))
)

process.electronMcSignalValidatorMiniAOD.InputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD")
process.electronMcSignalValidatorMiniAOD.OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD")

#process.p = cms.Path(process.ElectronIsolation * process.electronMcSignalValidatorMiniAOD * process.MEtoEDMConverter ) # * process.dqmStoreStats 
#process.p = cms.Path(process.electronMcSignalValidatorMiniAOD * process.MEtoEDMConverter * process.dqmStoreStats)
process.p = cms.Path( process.miniAODElectronIsolation * process.ElectronIsolation * process.electronMcSignalValidatorMiniAOD * process.MEtoEDMConverter ) #  process.printContent * 
process.outpath = cms.EndPath(
process.EDM,
)
