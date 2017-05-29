
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

electronMcSignalHistosCfg = cms.PSet(
  EfficiencyFlag = cms.bool(True), StatOverflowFlag = cms.bool(False)
)

electronMcSignalPostValidator = DQMEDHarvester("ElectronMcSignalPostValidator",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtJobEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcSignalValidator"),
  OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidator"),
    
  histosCfg = cms.PSet(electronMcSignalHistosCfg)
)



