import FWCore.ParameterSet.Config as cms

hgcGeomCheck = cms.EDAnalyzer("HGCGeometryCheck",
                              geometrySource = cms.untracked.vstring(
        'HGCalEESensitive',
        'HGCalHESiliconSensitive',
        'Hcal'),
                              g4Source = cms.InputTag("g4SimHits","HGCalInfoLayer"),
                              )
