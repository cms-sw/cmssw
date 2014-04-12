
import FWCore.ParameterSet.Config as cms

electronMcFakePostValidator = cms.EDAnalyzer("ElectronMcFakePostValidator",

  Verbosity = cms.untracked.int32(0),
  FinalStep = cms.string("AtRunEnd"),
  InputFile = cms.string(""),
  OutputFile = cms.string(""),
  InputFolderName = cms.string("EgammaV/ElectronMcFakeValidator"),
  OutputFolderName = cms.string("EgammaV/ElectronMcFakeValidator"),
    
)




