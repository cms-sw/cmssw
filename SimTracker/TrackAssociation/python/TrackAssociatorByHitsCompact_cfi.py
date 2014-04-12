import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import TrackAssociatorByHits
TrackAssociatorByHitsCompact = TrackAssociatorByHits.clone(
    ComponentName = cms.string('TrackAssociatorByHitsCompact'),
    useCompactStripLinks = cms.bool(True),
)
