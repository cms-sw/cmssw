import FWCore.ParameterSet.Config as cms
from SimG4Core.Configuration.SimG4Core_cff import *

g4SimHits.Watchers = cms.VPSet(cms.PSet(
        HGCalTBMB = cms.PSet(
            DetectorNames = cms.vstring(
                'HGCCerenkov',
                'HGCMTS6SC1',
                'HGCTelescope',
                'HGCMTS6SC2',  
                'HGCMTS6SC3',  
                'HGCHeTube',
                'HGCFeChamber',
                'HGCScint1',
                'HGCScint2',
                'HGCFSiTrack',
                'HGCAlPlate',
                'HGCalExtra',
                ),
            MaximumZ = cms.double(200.),
            StopName = cms.string("Junk"),
            ),
        type = cms.string('HGCalTBMB')
        )
                               )
