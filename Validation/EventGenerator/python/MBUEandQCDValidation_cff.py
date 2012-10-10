import FWCore.ParameterSet.Config as cms

from Validation.EventGenerator.MBUEandQCDValidation_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

chargedParticles = cms.EDFilter("GenParticleSelector",
    filter = cms.bool(False),
    src    = cms.InputTag("genParticles"),
    cut    = cms.string('charge != 0 & pt > 0.05 & status = 1 & eta < 2.5 & eta > -2.5')
)

chargedak5GenJets = ak5GenJets.clone( src = cms.InputTag("chargedParticles") )

mbueAndqcd_seq = cms.Sequence(chargedParticles*chargedak5GenJets*mbueAndqcdValidation)
