import FWCore.ParameterSet.Config as cms

# magnetic field
# cms geometry

# tracker geometry
# tracker numbering
# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import *
# PropagatorWithMaterialESProducer
#include "TrackingTools/MaterialEffects/data/OppositeMaterialPropagator.cfi"
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *

import RecoTracker.TrackProducer.TrackProducer_cfi
# TrackProducer
ctfNoOverlaps = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'ckfTrackCandidatesNoOverlaps'
)
