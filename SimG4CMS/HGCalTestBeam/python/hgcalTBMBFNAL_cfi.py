import FWCore.ParameterSet.Config as cms
from SimG4Core.Configuration.SimG4Core_cff import *

g4SimHits.Watchers = cms.VPSet(cms.PSet(
        HGCalTBMB = cms.PSet(
            DetectorNames = cms.vstring(
                'HGCMTS6SC3b',  
                'HGCHeTube',
                'HGCFeChamber',
                'HGCScint1',
                'HGCScint2',
                'HGCFSiTrack',
                'HGCAlPlate',
                ),
            MaximumZ = cms.double(200.),
            StopName = cms.string("HGCal"),
            ),
        type = cms.string('HGCalTBMB')
        )
                               )
