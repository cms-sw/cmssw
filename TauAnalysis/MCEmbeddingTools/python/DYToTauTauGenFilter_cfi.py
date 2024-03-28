import FWCore.ParameterSet.Config as cms


dyToTauTauGenFilter = cms.EDFilter("DYToTauTauGenFilter", 
                    inputTag = cms.InputTag("genParticles"),
                    filter = cms.bool(True))
