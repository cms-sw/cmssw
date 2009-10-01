
import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.ElectronMcStdBiningParameters_cff import *
from Validation.RecoEgamma.ElectronMcFineBiningParameters_cff import *

electronMcValidator = cms.EDAnalyzer("ElectronMcValidator",
  electronCollection = cms.InputTag("gsfElectrons"),
  mcTruthCollection = cms.InputTag("genParticles"),
  readAOD = cms.bool(False),
  outputFile = cms.string(""),
  MaxPt = cms.double(100.0),
  DeltaR = cms.double(0.05),
  MatchingID = cms.vint32(11,-11),
  MatchingMotherID = cms.vint32(23,24,-24,32),
  MaxAbsEta = cms.double(2.5),
  HistosConfigurationMC = cms.PSet(
    ElectronMcStdBiningParameters
    #ElectronMcFineBiningParameters
  )
)



