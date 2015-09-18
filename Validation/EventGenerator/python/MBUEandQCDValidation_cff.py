import FWCore.ParameterSet.Config as cms

from Validation.EventGenerator.MBUEandQCDValidation_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

chargedParticles = cms.EDFilter("GenParticleSelector",
    filter = cms.bool(False),
    src    = cms.InputTag("genParticles"),
    cut    = cms.string('obj.charge() != 0 && obj.pt() > 0.05 && obj.status() == 1 && std::abs(obj.eta()) < 2.5')
)

chargedak4GenJets = ak4GenJets.clone( src = cms.InputTag("chargedParticles") )

mbueAndqcd_seq = cms.Sequence(cms.ignore(chargedParticles)*chargedak4GenJets*mbueAndqcdValidation)
