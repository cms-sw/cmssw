import FWCore.ParameterSet.Config as cms

tauValidation = cms.EDAnalyzer("TauValidation",
    hepmcCollection = cms.InputTag("generator",""),
    tauEtCutForRtau = cms.double(50),
    UseWeightFromHepMC = cms.bool(True)
)
