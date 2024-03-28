import FWCore.ParameterSet.Config as cms


dyToElElGenFilter = cms.EDFilter("DYToElElGenFilter", 
                    inputTag = cms.InputTag("genParticles"),
                    #filter = cms.bool(True)
                    )
