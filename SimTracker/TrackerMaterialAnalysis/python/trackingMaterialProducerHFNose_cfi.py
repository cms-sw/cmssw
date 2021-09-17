import FWCore.ParameterSet.Config as cms

import copy
from SimG4Core.Application.g4SimHits_cfi import *
# Detector simulation (Geant4-based)
trackingMaterialProducer = copy.deepcopy(g4SimHits)
trackingMaterialProducer.Generator.HepMCProductLabel = 'generatorSmeared'

#trackingMaterialProducer.Physics.type = 'SimG4Core/Physics/DummyPhysics'
#trackingMaterialProducer.Physics.DummyEMPhysics = True
#trackingMaterialProducer.Physics.CutsPerRegion = False
trackingMaterialProducer.UseMagneticField = False

trackingMaterialProducer.Watchers = cms.VPSet(cms.PSet(
    TrackingMaterialProducer = cms.PSet(
        PrimaryTracksOnly = cms.bool(True),
        #The file to direct the HGCal volumes z position
        txtOutFile = cms.untracked.string('VolumesZPositionHFNose.txt'),
        #In the beginning of each track, the track will first hit SS and it will 
        #save the upper z volume boundary. So, the low boundary of the first SS 
        #volume is never saved. Here we give the low boundary. 
        #This can be found by running
        #Geometry/HGCalCommonData/test/testHGCalParameters_cfg.py
        #on the geometry under study and looking for zFront print out. 
        #Plus or minus endcap is being dealt with inside the code. 
        hgcalzfront = cms.double(10484.0),
        SelectedVolumes = cms.vstring('HFNose')
    ),
    type = cms.string('TrackingMaterialProducer')
))
