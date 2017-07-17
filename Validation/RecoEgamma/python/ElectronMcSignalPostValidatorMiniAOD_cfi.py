
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

electronMcSignalHistosCfg = cms.PSet(
  EfficiencyFlag = cms.bool(True), StatOverflowFlag = cms.bool(False)
)

electronMcSignalPostValidatorMiniAOD = DQMEDHarvester("ElectronMcSignalPostValidatorMiniAOD",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtJobEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD"),
  OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD"),
    
  histosCfg = cms.PSet(electronMcSignalHistosCfg)
)



