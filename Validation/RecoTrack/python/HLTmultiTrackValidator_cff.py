import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff import *

hltTrackValidator = hltMultiTrackValidator.clone(
    label = [
        "hltPixelTracks",
        "hltMergedTracks"
    ]
)

hltMultiTrackValidationTask = cms.Task(
    hltTPClusterProducer
    , trackingParticleNumberOfLayersProducer
    , hltTrackAssociatorByHits
)
hltMultiTrackValidation = cms.Sequence(
    hltTrackValidator,
    hltMultiTrackValidationTask
)
