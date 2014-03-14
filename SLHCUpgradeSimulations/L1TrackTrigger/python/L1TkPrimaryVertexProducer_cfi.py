import FWCore.ParameterSet.Config as cms

L1TkPrimaryVertex = cms.EDProducer('L1TkPrimaryVertexProducer',
     L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
     ZMAX = cms.double ( 25. ) ,        # in cm
     CHI2MAX = cms.double( 100. ),
     DeltaZ = cms.double( 0.1 ),        # in cm.   
     PTMINTRA = cms.double( 2.),        # PTMIN of L1Tracks, in GeV
     nStubsmin = cms.int32( 4 ) ,       # minimum number of stubs
     nStubsPSmin = cms.int32( 3 ),      # minimum number of stubs in PS modules 
     SumPtSquared = cms.bool( True )    # maximizes SumPT^2 or SumPT
)

