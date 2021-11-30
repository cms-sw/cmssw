import FWCore.ParameterSet.Config as cms

cscRecHitPSet = dict(
    cscRecHit = dict(
        verbose = 0,
        inputTag = "csc2DRecHits",
        minBX = -1,
        maxBX = 1,
    ),
    cscSegment = dict(
        verbose = 0,
        inputTag = "cscSegments",
        minBX = -1,
        maxBX = 1,
    )
)
