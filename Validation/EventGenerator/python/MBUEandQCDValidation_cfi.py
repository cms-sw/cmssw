import FWCore.ParameterSet.Config as cms

mbueAndqcdValidation = cms.EDAnalyzer("MBUEandQCDValidation",
    hepmcCollection = cms.InputTag("generator",""),
    genChjetsCollection = cms.InputTag("chargedak4GenJetsNoNu",""),
    genjetsCollection = cms.InputTag("ak4GenJetsNoNu",""),
    verbosity = cms.untracked.uint32(0),
    UseWeightFromHepMC = cms.bool(True)
)
