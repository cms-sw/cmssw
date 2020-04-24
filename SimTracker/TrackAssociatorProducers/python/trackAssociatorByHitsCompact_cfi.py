import FWCore.ParameterSet.Config as cms

import SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi as tabh
TrackAssociatorByHitsCompact = tabh.trackAssociatorByHits.clone(
    useCompactStripLinks = cms.bool(True)
)
