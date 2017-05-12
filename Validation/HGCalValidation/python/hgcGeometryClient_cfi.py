import FWCore.ParameterSet.Config as cms

hgcalGeometryClient = cms.EDProducer("HGCalGeometryClient", 
                                     DirectoryName = cms.string("Geometry"),
                                     )
