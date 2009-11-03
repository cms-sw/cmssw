import FWCore.ParameterSet.Config as cms


from DQMOffline.PFTau.pfCandidateManager_cfi import pfCandidateManager

pfCandidateManagerMatch = pfCandidateManager.clone()
pfCandidateManagerMatch.MatchCollection = 'genParticles'

pfCandidateManagerSequence = cms.Sequence( pfCandidateManagerMatch )
