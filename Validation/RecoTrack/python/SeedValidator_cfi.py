import FWCore.ParameterSet.Config as cms

seedValidator = cms.EDFilter("SeedValidator",
    associators = cms.vstring('TrackAssociatorByHits'),
    useFabsEta = cms.bool(True),
    minpT = cms.double(0.0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    min = cms.double(0.0),
    max = cms.double(2.5),
    nintHit = cms.int32(25),
    label = cms.VInputTag(cms.InputTag("globalMixedSeeds")),
    maxHit = cms.double(25.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    nintpT = cms.int32(200),
    label_tp_fake = cms.InputTag("cutsTPFake"),
    label_tp_effic = cms.InputTag("cutsTPEffic"),
    useInvPt = cms.bool(False),
    maxpT = cms.double(100.0),
    out = cms.string('validationPlotsSeed.root'),
    minHit = cms.double(0.0),
    sim = cms.string('g4SimHits'),
    nint = cms.int32(25)
)


