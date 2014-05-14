import FWCore.ParameterSet.Config as cms

L1TkPrimaryVertex = cms.EDProducer('L1TkFastVertexProducer',
     L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
     ZMAX = cms.double ( 25. ) ,        # in cm
     CHI2MAX = cms.double( 100. ),
     PTMINTRA = cms.double( 2.),        # PTMIN of L1Tracks, in GeV
     nStubsmin = cms.int32( 4 ) ,       # minimum number of stubs
     nStubsPSmin = cms.int32( 3 ),       # minimum number of stubs in PS modules 
     PTMAX = cms.double( 50. ),          # in GeV. When PTMAX > 0, tracks with PT above PTMAX are considered as
					 # mismeasured and are treated according to HighPtTracks below.
					 # When PTMAX < 0, no special treatment is done for high PT tracks.
					 # If PTMAX < 0, no saturation or truncation is done.
     HighPtTracks = cms.int32( 1 )	 # when = 0 : truncation. Tracks with PT above PTMAX are ignored 
					 # when = 1 : saturation. Tracks with PT above PTMAX are set to PT=PTMAX.
)
