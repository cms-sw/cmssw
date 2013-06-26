
import sys
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronWget")

process.DQMStore = cms.Service("DQMStore")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# input file
import DQMOffline.EGamma.electronDataDiscovery as dd
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source.fileNames.extend(dd.search())

process.load("Validation.RecoEgamma.ElectronWebGetter_cfi")
#process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")

process.electronWebGetter.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
process.electronWebGetter.InputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")
process.electronWebGetter.OutputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")

#from DQMServices.Components.EDMtoMEConverter_cff import *
#EDMtoMEConverter.Verbosity = 0
#EDMtoMEConverter.convertOnEndRun = True
#EDMtoME = cms.Sequence(EDMtoMEConverter)

#process.p = cms.Path(EDMtoME*process.electronWebGetter)
process.p = cms.Path(process.electronWebGetter)


