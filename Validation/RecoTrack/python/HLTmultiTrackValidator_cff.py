import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff import *

hltTrackValidator = hltMultiTrackValidator.clone(
    label = [
        "hltPixelTracks",
        "hltIter0PFlowTrackSelectionHighPurity",
        "hltIter1PFlowTrackSelectionHighPurity",
        "hltIter1Merged",
        "hltIter2PFlowTrackSelectionHighPurity",
        "hltIter2Merged",
        "hltTracksMerged",
#        "hltIter3PFlowTrackSelectionHighPurity",
#        "hltIter3Merged",
#        "hltIter4PFlowTrackSelectionHighPurity",
#        "hltIter4Merged",
    ]
)

hltMultiTrackValidation = cms.Sequence(
    hltTPClusterProducer
    + trackingParticleNumberOfLayersProducer
    + hltTrackAssociatorByHits
    + hltTrackValidator
)
