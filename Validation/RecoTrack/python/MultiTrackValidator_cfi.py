import FWCore.ParameterSet.Config as cms

multiTrackValidator = cms.EDFilter("MultiTrackValidator",
    associators = cms.vstring('TrackAssociatorByHits', 'TrackAssociatorByChi2'),
    useFabsEta = cms.bool(True),
    minpT = cms.double(0.0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    min = cms.double(0.0),
    max = cms.double(2.5),
    nintHit = cms.int32(25),
    label = cms.VInputTag(cms.InputTag("cutsRecoTracks")),
    maxHit = cms.double(25.0),
    nintpT = cms.int32(200),
    label_tp_fake = cms.InputTag("cutsTPFake"),
    label_tp_effic = cms.InputTag("cutsTPEffic"),
    useInvPt = cms.bool(False),
    maxpT = cms.double(100.0),
    out = cms.string('validationPlots.root'),
    minHit = cms.double(0.0),
    sim = cms.string('g4SimHits'),
    nint = cms.int32(25)
)


