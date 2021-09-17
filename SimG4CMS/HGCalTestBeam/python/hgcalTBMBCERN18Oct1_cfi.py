import FWCore.ParameterSet.Config as cms
from SimG4Core.Configuration.SimG4Core_cff import *

g4SimHits.Watchers = cms.VPSet(cms.PSet(
    HGCalTBMB = cms.PSet(
        DetectorNames = cms.vstring(
            'HGCalBeamWChamb',  
            'HGCalBeamS1',
            'HGCalBeamS2',
            'HGCalBeamS3',
            'HGCalBeamS4',
            'HGCalBeamS5',
            'HGCalBeamS6',
            'HGCalBeamCK3',
            'HGCalBeamHaloCounter',
            'HGCalBeamMuonCounter',
            'HGCalEE',
            'HGCalHE',
            'HGCalAH'
        ),
        MaximumZ = cms.double(15000.0),
        StopName = cms.string("Junk"),
    ),
    type = cms.string('HGCalTBMB')
))
