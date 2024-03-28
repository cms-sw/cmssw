import FWCore.ParameterSet.Config as cms


dyToElTauGenFilter = cms.EDFilter("DYToElTauGenFilter", 
                    inputTag = cms.InputTag("genParticles"),
                    filter = cms.bool(True))
