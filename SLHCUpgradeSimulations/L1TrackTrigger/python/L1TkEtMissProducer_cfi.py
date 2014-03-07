import FWCore.ParameterSet.Config as cms

L1TkEtMiss = cms.EDProducer('L1TkEtMissProducer',
     L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
     L1VertexInputTag = cms.InputTag("L1TkPrimaryVertex"),
     ZMAX = cms.double ( 25. ) ,        # in cm
     CHI2MAX = cms.double( 100. ),
     PTMINTRA = cms.double( 2. ),       # in GeV
     DeltaZ = cms.double( 1. ),       # in cm
     nStubsmin = cms.int32( 4 ),      # min number of stubs for the tracks to enter in TrkMET calculation
     nStubsPSmin = cms.int32( 0)      # min number of stubs in the PS Modules
)


