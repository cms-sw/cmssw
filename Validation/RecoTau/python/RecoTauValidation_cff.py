import FWCore.ParameterSet.Config as cms

from Validation.RecoTau.RecoTauValidationMiniAOD_cfi import *
tauValidationMiniAODZTT = tauValidationMiniAOD.clone()
tauValidationMiniAODZEE = tauValidationMiniAOD.clone(
  RefCollection = cms.InputTag("kinematicSelectedTauValDenominatorZEE"),
  ExtensionName = cms.string('ZEE')
)
tauValidationMiniAODZMM = tauValidationMiniAOD.clone(
  RefCollection = cms.InputTag("kinematicSelectedTauValDenominatorZMM"),
  ExtensionName = cms.string('ZMM')
)
tauValidationMiniAODQCD = tauValidationMiniAOD.clone(
  RefCollection = cms.InputTag("kinematicSelectedTauValDenominatorQCD"),
  ExtensionName = cms.string('QCD')
)
tauValidationMiniAODRealData = tauValidationMiniAOD.clone(
  RefCollection = cms.InputTag("CleanedPFJets"),
  ExtensionName = cms.string('RealData')
)
tauValidationMiniAODRealElectronsData = tauValidationMiniAOD.clone(
  RefCollection = cms.InputTag("ElZLegs","theProbeLeg"),
  ExtensionName = cms.string("RealElectronsData")
)
tauValidationMiniAODRealMuonsData = tauValidationMiniAOD.clone(
  RefCollection = cms.InputTag("MuZLegs","theProbeLeg"),
  ExtensionName = cms.string('RealMuonsData')
)

tauValidationSequenceMiniAOD = cms.Sequence(tauValidationMiniAODZTT*tauValidationMiniAODZEE*tauValidationMiniAODZMM*tauValidationMiniAODQCD*tauValidationMiniAODRealData*tauValidationMiniAODRealElectronsData*tauValidationMiniAODRealMuonsData)
