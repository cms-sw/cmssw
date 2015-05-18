import FWCore.ParameterSet.Config as cms

l1TkMuonsExt = cms.EDProducer(
    "L1TkMuonFromExtendedProducer",
    L1MuonsInputTag = cms.InputTag("l1extraMuExtended"),
    L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis", "Level1TTTracks"),
    ETAMIN = cms.double(0),
    ETAMAX = cms.double(5.),        # no cut
    ZMAX = cms.double( 25. ),       # in cm
    CHI2MAX = cms.double( 100. ),
    PTMINTRA = cms.double( 2. ),    # in GeV
#    DRmax = cms.double( 0.5 ),
    nStubsmin = cms.int32( 3 ),        # minimum number of stubs
#    closest = cms.bool( True ),
    correctGMTPropForTkZ = cms.bool(True),
    use5ParameterFit = cms.bool(False) #use 4-pars by defaults
    
)


