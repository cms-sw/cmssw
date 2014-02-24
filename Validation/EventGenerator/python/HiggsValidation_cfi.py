import FWCore.ParameterSet.Config as cms

Z0testValidation = cms.EDAnalyzer("HiggsValidation",
    hepmcCollection    = cms.InputTag("generator",""),
    pdg_id             = cms.int32(25),
    particleName       = cms.string("Higgs"),
    monitorDecays      = cms.vstring("tau+tau","mu+mu","e+e","b+b","gamma+gamma","W+W","Z+Z","Z+gamma"),
    UseWeightFromHepMC = cms.bool(True)
)

hplusValidation = cms.EDAnalyzer("HiggsValidation",
    hepmcCollection    = cms.InputTag("generator",""),
    pdg_id             = cms.int32(37),
    particleName       = cms.string("Hplus"),
    monitorDecays      = cms.vstring("tau+taunu","mu+munu","e+enu","t+b","c+s"),
    UseWeightFromHepMC = cms.bool(True)
)

higgsValidation = cms.Sequence(
    Z0testValidation*
    hplusValidation
)
