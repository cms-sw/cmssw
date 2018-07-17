import FWCore.ParameterSet.Config as cms
from SimG4Core.Configuration.SimG4Core_cff import *

g4SimHits.Watchers = cms.VPSet(cms.PSet(
        SimG4HGCalValidation = cms.PSet(
            Names = cms.vstring(
                'HGCalEECell',  
                'HGCalHECell',
                'HEScintillator',
                ),
            Types = cms.vint32(1,1,2),
            LabelLayerInfo = cms.string("HGCalInfoLayer"),
            ),
        type = cms.string('SimG4HGCalValidation')
        )
                               )

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hgcGeomAnalysis = DQMEDAnalyzer('HGCGeometryValidation',
                                 geometrySource = cms.untracked.vstring(
        'HGCalEESensitive',
        'HGCalHESiliconSensitive',
        'Hcal'),
                                 g4Source = cms.InputTag("g4SimHits","HGCalInfoLayer"),
                                 )
