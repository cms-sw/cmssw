import FWCore.ParameterSet.Config as cms

hgcalGeometryClient = cms.EDAnalyzer("HGCalGeometryClient", 
                                     DirectoryName = cms.string("Geometry"),
                                     )
