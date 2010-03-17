
import FWCore.ParameterSet.Config as cms

trackTriggerHits = cms.EDProducer("TrackTriggerHitsFromMC",
    doPileUp = cms.bool(True),
    magField = cms.double(4.0),
    inputTag = cms.InputTag("mix")
)

