import FWCore.ParameterSet.Config as cms

L1TkMuonsNaive = cms.EDProducer("L1TkMuonNaiveProducer",
        L1MuonsInputTag = cms.InputTag("l1extraParticles"),
        L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
        ETAMIN = cms.double(0),
        ETAMAX = cms.double(5.),        # no cut
        ZMAX = cms.double( 25. ),       # in cm
        CHI2MAX = cms.double( 100. ),
        PTMINTRA = cms.double( 2. ),    # in GeV
        DRmax = cms.double( 0.5 ),
        nStubsmin = cms.int32( 5 ),        # minimum number of stubs
        closest = cms.bool( True )
)

