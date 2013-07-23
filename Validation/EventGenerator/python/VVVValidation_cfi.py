import FWCore.ParameterSet.Config as cms

vvvValidation = cms.EDAnalyzer("VVVValidation",
    hepmcCollection = cms.InputTag("generator",""),
    genparticleCollection = cms.InputTag("genParticles",""),
    genjetsCollection = cms.InputTag("ak5GenJets",""),
    matchingPrecision = cms.double(0.001),
    lepStatus= cms.double(1),
    motherStatus= cms.double(1),
    verbosity = cms.untracked.uint32(0),
    UseWeightFromHepMC = cms.bool(True)
)
