
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

electronMcFakeHistosCfg = cms.PSet(
  EfficiencyFlag = cms.bool(True),StatOverflowFlag = cms.bool(False)
)

electronMcFakePostValidator = DQMEDHarvester("ElectronMcFakePostValidator",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtJobEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcFakeValidator"),
  OutputFolderName = cms.string("EgammaV/ElectronMcFakeValidator"),
    
  histosCfg = cms.PSet(electronMcFakeHistosCfg)
)




