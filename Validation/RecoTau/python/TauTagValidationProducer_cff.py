import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets

#from Validation.RecoParticleFlow.tauGenJetForEWKTauSelector_cfi import *

tauGenJetProducer = cms.Sequence(
    tauGenJets 
#    selectedGenTauDecaysToHadrons
    )
