import FWCore.ParameterSet.Config as cms

import copy
from SimG4Core.Application.g4SimHits_cfi import *
# Detector simulation (Geant4-based)
trackingMaterialProducer = copy.deepcopy(g4SimHits)
trackingMaterialProducer.Generator.HepMCProductLabel = 'generatorSmeared'
trackingMaterialProducer.Physics.type = 'SimG4Core/Physics/DummyPhysics'
trackingMaterialProducer.Physics.DummyEMPhysics = True
trackingMaterialProducer.Physics.CutsPerRegion = False
trackingMaterialProducer.UseMagneticField = False
trackingMaterialProducer.Watchers = cms.VPSet(cms.PSet(
    TrackingMaterialProducer = cms.PSet(
        PrimaryTracksOnly = cms.bool(True),
        #The following variables are set for HGCal support and 
        #it doesn't effect or run at all for Tracker
        txtOutFile = cms.untracked.string('VolumesZPosition.txt'),
        hgcalzfront = cms.double(3190.5),
        SelectedVolumes = cms.vstring('BEAM', 
            'Tracker')
    ),
    type = cms.string('TrackingMaterialProducer')
))
