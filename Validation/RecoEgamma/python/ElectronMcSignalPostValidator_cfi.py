
import FWCore.ParameterSet.Config as cms

electronMcSignalHistosCfg = cms.PSet(
  setEfficiencyFlag = cms.bool(True)
)

electronMcSignalPostValidator = cms.EDAnalyzer("ElectronMcSignalPostValidator",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtRunEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcSignalValidator"),
  OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidator"),
    
  histosCfg = cms.PSet(electronMcSignalHistosCfg)
)



