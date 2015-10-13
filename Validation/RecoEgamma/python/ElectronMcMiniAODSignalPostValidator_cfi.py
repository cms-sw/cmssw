
import FWCore.ParameterSet.Config as cms

electronMcSignalHistosCfg = cms.PSet(
  EfficiencyFlag = cms.bool(True), StatOverflowFlag = cms.bool(False)
)

electronMcMiniAODSignalPostValidator = cms.EDAnalyzer("ElectronMcMiniAODSignalPostValidator",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtJobEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcMiniAODSignalValidator"),
  OutputFolderName = cms.string("EgammaV/ElectronMcMiniAODSignalValidator"),
    
  histosCfg = cms.PSet(electronMcSignalHistosCfg)
)



