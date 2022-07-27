import FWCore.ParameterSet.Config as cms

## Load propagator
from TrackPropagation.Geant4e.Geant4ePropagator_cfi import *


from TrackingTools.TrackRefitter.TracksToTrajectories_cff import *

from SimG4Core.Application.g4SimHits_cfi import g4SimHits as _g4SimHits


## Set up geometry
geopro = cms.EDProducer("GeometryProducer",
     GeoFromDD4hep = cms.bool(False),
     UseMagneticField = cms.bool(True),
     UseSensitiveDetectors = cms.bool(False),
     MagneticField =  _g4SimHits.MagneticField.clone()
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toModify(geopro, GeoFromDD4hep = True )


# load this to do a track refit
from RecoTracker.TrackProducer.TrackRefitters_cff import *
from RecoVertex.V0Producer.generalV0Candidates_cff import *
from TrackPropagation.Geant4e.Geant4ePropagator_cfi import *

G4eFitter = RKTrajectoryFitter.clone ( 
    ComponentName = cms.string('G4eFitter'),
    Propagator = cms.string('Geant4ePropagator') 
)

G4eSmoother = RKTrajectorySmoother.clone (
     ComponentName = cms.string('G4eSmoother'),
     Propagator = cms.string('Geant4ePropagator'),
 
     ## modify rescaling to have a more stable fit during the backward propagation
     errorRescaling = cms.double(2.0)
)

G4eFitterSmoother = KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = cms.string('G4eFitterSmoother'),
    Fitter = cms.string('G4eFitter'),
    Smoother = cms.string('G4eSmoother'),
    ## will reject no hits
    EstimateCut = cms.double(-1.0)
)

## Different versions have different refitter cffs
from RecoTracker.TrackProducer.TrackRefitters_cff import *

# configure the refitter with the G4e
# automatically uses the generalTracks collection as input
Geant4eTrackRefitter = TrackRefitter.clone()
Geant4eTrackRefitter.Fitter = cms.string('G4eFitterSmoother')
Geant4eTrackRefitter.Propagator = cms.string('Geant4ePropagator')

geant4eTrackRefit = cms.Sequence(geopro*Geant4eTrackRefitter)
