
import FWCore.ParameterSet.Config as cms

electronMcSignalHistosCfg = cms.PSet(
  EfficiencyFlag = cms.bool(True), StatOverflowFlag = cms.bool(False)
)

electronMcSignalPostValidatorMiniAOD = cms.EDAnalyzer("ElectronMcSignalPostValidatorMiniAOD",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtJobEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD"),
  OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorMiniAOD"),
    
  histosCfg = cms.PSet(electronMcSignalHistosCfg)
)



