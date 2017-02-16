import FWCore.ParameterSet.Config as cms


dYToMuMuGenFilter = cms.EDFilter("DYToMuMuGenFilter", 
                              inputTag = cms.InputTag("prunedGenParticles"))
