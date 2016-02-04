import FWCore.ParameterSet.Config as cms

## Load propagator
from TrackPropagation.Geant4e.Geant4ePropagator_cfi import *

## Set up geometry
geopro = cms.EDProducer("GeometryProducer",
     UseMagneticField = cms.bool(True),
     UseSensitiveDetectors = cms.bool(False),
     MagneticField = cms.PSet(
         UseLocalMagFieldManager = cms.bool(False),
         Verbosity = cms.untracked.bool(False),
         ConfGlobalMFM = cms.PSet(
             Volume = cms.string('OCMS'),
             OCMS = cms.PSet(
                 Stepper = cms.string('G4ClassicalRK4'),
                 Type = cms.string('CMSIMField'),
                 G4ClassicalRK4 = cms.PSet(
                     MaximumEpsilonStep = cms.untracked.double(0.01), ## in mm
                     DeltaOneStep = cms.double(0.001), ## in mm
                     MaximumLoopCounts = cms.untracked.double(1000.0),
                     DeltaChord = cms.double(0.001), ## in mm
                     MinStep = cms.double(0.1), ## in mm
                     DeltaIntersectionAndOneStep = cms.untracked.double(-1.0),
                     DeltaIntersection = cms.double(0.0001), ## in mm
                     MinimumEpsilonStep = cms.untracked.double(1e-05) ## in mm
                 )
             )
         ),
         delta = cms.double(1.0)
     )

   )


## Create G4e fitter - smoothing doesn't work with current Geant release
##    working on getting it added
G4eFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
                           ComponentName = cms.string('G4eFitter'),
                           Estimator = cms.string('Chi2'),
                           Propagator = cms.string('Geant4ePropagator'),
                           Updator = cms.string('KFUpdator'),
                           minHits = cms.int32(3)
                           )

## Different versions have different refitter cffs
from RecoTracker.TrackProducer.TrackRefitters_cff import *
#from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
Geant4eRefitter = TrackRefitter.clone()
Geant4eRefitter.Propagator=cms.string("Geant4ePropagator")
Geant4eRefitter.Fitter=cms.string("G4eFitter")

geant4eTrackRefit = cms.Sequence(geopro*Geant4eRefitter)
