import FWCore.ParameterSet.Config as cms

L1TkEtMiss = cms.EDProducer('L1TkEtMissProducer',
     L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
     L1VertexInputTag = cms.InputTag("L1TkPrimaryVertex"),
     ZMAX = cms.double ( 25. ) ,        # in cm
     CHI2MAX = cms.double( 100. ),
     PTMINTRA = cms.double( 2. ),       # in GeV
     DeltaZ = cms.double( 1. ),       # in cm
     nStubsmin = cms.int32( 4 ),      # min number of stubs for the tracks to enter in TrkMET calculation
     nStubsPSmin = cms.int32( 0),      # min number of stubs in the PS Modules
     #PTMAX = cms.double( 50. ),	       # in GeV. When PTMAX > 0, tracks with PT above PTMAX are considered as
                                         # mismeasured and are treated according to HighPtTracks below.
                                         # When PTMAX < 0, no special treatment is done for high PT tracks.
     PTMAX = cms.double( 50. ),
     HighPtTracks = cms.int32( 0 ),       # when = 0 : truncation. Tracks with PT above PTMAX are ignored 
                                         # when = 1 : saturation. Tracks with PT above PTMAX are set to PT=PTMAX.
                                         # When PTMAX < 0, no special treatment is done for high PT tracks.
     doPtComp = cms.bool( True ),       # track-stubs PT compatibility cut
     doTightChi2 = cms.bool( False )    # chi2dof < 5 for tracks with PT > 10


)


