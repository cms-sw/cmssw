import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets
from Validation.RecoParticleFlow.pfTauBenchmarkElecRejection_cfi import pfTauBenchmarkElecRejection


tauBenchmarkElecRejection = cms.Sequence(
    pfTauBenchmarkElecRejection
)
