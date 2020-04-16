import FWCore.ParameterSet.Config as cms
from SimG4Core.Configuration.SimG4Core_cff import *

g4SimHits.Watchers = cms.VPSet(cms.PSet(
    HGCalTBMB = cms.PSet(
        DetectorNames = cms.vstring(
            'HGCalBeamWChamb',  
            'HGCalBeamAl1',
            'HGCalBeamAl2',
            'HGCalBeamAl3',
            'HGCalBeamTube1',
            'HGCalBeamTube2',
            'HGCalBeamTube3',
            'HGCalEE',
            'HGCalHE',
            'HGCalAH'
        ),
        MaximumZ = cms.double(15000.),
        StopName = cms.string("Junk"),
    ),
    type = cms.string('HGCalTBMB')
))
