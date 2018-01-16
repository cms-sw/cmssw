import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
Z0testValidation = DQMEDAnalyzer('HiggsValidation',
    hepmcCollection    = cms.InputTag("generatorSmeared"),
    pdg_id             = cms.int32(25),
    particleName       = cms.string("Higgs"),
    monitorDecays      = cms.vstring("tau+tau","mu+mu","e+e","b+b","gamma+gamma","W+W","Z+Z","Z+gamma"),
    UseWeightFromHepMC = cms.bool(True)
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hplusValidation = DQMEDAnalyzer('HiggsValidation',
    hepmcCollection    = cms.InputTag("generatorSmeared"),
    pdg_id             = cms.int32(37),
    particleName       = cms.string("Hplus"),
    monitorDecays      = cms.vstring("tau+taunu","mu+munu","e+enu","t+b","c+s"),
    UseWeightFromHepMC = cms.bool(True)
)

higgsValidation = cms.Sequence(
    Z0testValidation*
    hplusValidation
)
