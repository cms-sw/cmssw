import FWCore.ParameterSet.Config as cms

mbueAndqcdValidation = cms.EDAnalyzer("MBUEandQCDValidation",
    hepmcCollection = cms.InputTag("generator",""),
    genChjetsCollection = cms.InputTag("chargedak4GenJets",""),
    genjetsCollection = cms.InputTag("ak4GenJets",""),
    verbosity = cms.untracked.uint32(0),
    UseWeightFromHepMC = cms.bool(True)
)
