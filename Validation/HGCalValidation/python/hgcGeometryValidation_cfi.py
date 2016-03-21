import FWCore.ParameterSet.Config as cms

hgcGeomAnalysis = cms.EDAnalyzer("HGCGeometryValidation",
                                 geometrySource = cms.untracked.vstring(
        'HGCalEESensitive',
        'HGCalHESiliconSensitive',
        'Hcal'),
                                 g4Source = cms.InputTag("g4SimHits","HGCalInfoLayer"),
                                 )
