
import FWCore.ParameterSet.Config as cms

electronMcSignalHistosCfg = cms.PSet(
  EfficiencyFlag = cms.bool(True), StatOverflowFlag = cms.bool(False)
)

electronMcSignalPostValidatorPt1000 = cms.EDAnalyzer("ElectronMcSignalPostValidator",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtJobEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorPt1000"),
  OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorPt1000"),
  
  histosCfg = cms.PSet(electronMcSignalHistosCfg)
)



