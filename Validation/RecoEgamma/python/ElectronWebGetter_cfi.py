
import FWCore.ParameterSet.Config as cms

electronWebGetter = cms.EDAnalyzer("ElectronWebGetter",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtRunEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcSignalValidator"),
  OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidator"),
    
)



