import FWCore.ParameterSet.Config as cms

L1TkTracksFromPixelDigis = cms.EDProducer("L1TkTrackBuilder_PixelDigi_",
      L1TkStubsBricks = cms.InputTag("L1TkStubsFromPixelDigis","StubsPass"),
      AssociativeMemories = cms.bool(False)
)



