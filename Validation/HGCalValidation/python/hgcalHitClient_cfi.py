import FWCore.ParameterSet.Config as cms

hgcalHitClient = cms.EDAnalyzer("HGCalHitClient", 
                                DirectoryName = cms.string("HitValidation"),
                                )
