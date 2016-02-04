import FWCore.ParameterSet.Config as cms


from DQMOffline.PFTau.pfCandidateManager_cfi import pfCandidateManager

pfCandidateManagerMatch = pfCandidateManager.clone()
pfCandidateManagerMatch.MatchCollection = 'genParticles'
pfCandidateManagerMatch.mode = 2


pfCandidateManagerSequence = cms.Sequence( pfCandidateManagerMatch )
