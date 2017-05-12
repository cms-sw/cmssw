import FWCore.ParameterSet.Config as cms

hgcalHitClient = cms.EDProducer("HGCalHitClient", 
                                DirectoryName = cms.string("HitValidation"),
                                )
