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
            'HGCalMCPAluminium',
            'HGCalMCPPyrexGlass',
            'HGCalMCPLeadGlass',
        ),
        MaximumZ = cms.double(25000.),
        StopName = cms.string("HGCal"),
    ),
    type = cms.string('HGCalTBMB')
))
