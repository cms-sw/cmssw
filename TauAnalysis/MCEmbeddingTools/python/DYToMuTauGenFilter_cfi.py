import FWCore.ParameterSet.Config as cms


dyToMuTauGenFilter = cms.EDFilter("DYToMuTauGenFilter", 
                    inputTag = cms.InputTag("genParticles"),
                    filter = cms.bool(True))
