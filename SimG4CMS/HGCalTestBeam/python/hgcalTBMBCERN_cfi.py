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
            'HGCalBeamHaloCounter',
            'HGCalBeamCK3',
            'HGCalBeamMuonCounter',
            'HGCalEE',
            'HGCalHE',
            'HGCalAH'
        ),
        #MaximumZ = cms.double(9500.),
        MaximumZ = cms.double(950000.),
        #StopName = cms.string("HGCal"),
        StopName = cms.string("Junk"),
    ),
    type = cms.string('HGCalTBMB')
))
