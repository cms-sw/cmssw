import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets


tauGenJetProducer = cms.Sequence(
    tauGenJets
    )
