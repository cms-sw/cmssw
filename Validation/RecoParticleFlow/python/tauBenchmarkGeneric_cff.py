import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.pfTauBenchmarkGeneric_cfi import pfTauBenchmarkGeneric
from Validation.RecoParticleFlow.caloTauBenchmarkGeneric_cfi import caloTauBenchmarkGeneric
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets


tauBenchmarkGeneric = cms.Sequence(
    tauGenJets + 
    pfTauBenchmarkGeneric +
    caloTauBenchmarkGeneric
    )
