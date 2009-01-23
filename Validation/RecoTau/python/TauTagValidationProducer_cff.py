import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets

from Validation.RecoParticleFlow.tauGenJetForValidationTauSelector_cfi import *

tauGenJetProducer = cms.Sequence(
    tauGenJets + 
    selectedGenTauDecaysToHadrons +
    selectedGenTauDecaysToHadronsEta25Cumulative +
    selectedGenTauDecaysToHadronsPt5Cumulative
    )
