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
        txtOutFile = cms.untracked.string('VolumesZPosition.txt'),
        #In the beginning of each track, the track will first hit an HGCAL volume and it will
        #save the upper z volume boundary. So, the low boundary of the first
        #volume is never saved. Here we give the low boundary of the first volume.
        #This can be found by asking first to run not on 'HGCal' volume below but
        #on 'CALOECTSRear', which at the moment of this writing it contains
        #HGCalService, HGCal and thermal screen. You should run Fireworks to
        #check if these naming conventions and volumes are valid in the future.
        #Then, check the VolumesZPosition.txt file to see where CEService ends and
        #put that number in hgcalzfront below. Keep in mind to run on the desired volume here:
        #https://github.com/cms-sw/cmssw/blob/master/SimTracker/TrackerMaterialAnalysis/plugins/TrackingMaterialProducer.cc#L95
        #and to replace the volume name of the material first hit at the file creation line:
        #https://github.com/cms-sw/cmssw/blob/master/SimTracker/TrackerMaterialAnalysis/plugins/TrackingMaterialProducer.cc#L159-L168
        hgcalzfront = cms.double(3210.5),
        SelectedVolumes = cms.vstring('HGCal')#CALOECTSRear HGCal
    ),
    type = cms.string('TrackingMaterialProducer')
))
