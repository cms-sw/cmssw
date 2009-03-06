import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

multiTrackValidator = cms.EDAnalyzer("MultiTrackValidator",
    dirName = cms.string('RecoMuonV/MultiTrack/'),
    out = cms.string(''),
    outputFile = cms.string(''),

    sim = cms.string('g4SimHits'),
    label_tp_effic = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label_tp_fake = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label = cms.VInputTag(cms.InputTag("globalMuons")),

    lipTP = cms.double(30.0),
    tipTP = cms.double(3.5),
    chargedOnlyTP = cms.bool(True),
    pdgIdTP = cms.vint32(13, -13),
    signalOnlyTP = cms.bool(True),
    minRapidityTP = cms.double(-2.4),
    maxRapidityTP = cms.double(2.4),
    minHitTP = cms.int32(0),
    ptMinTP = cms.double(0.9),

    UseAssociators = cms.bool(False),
    associators = cms.vstring('TrackAssociatorByDeltaR'),
    associatormap = cms.InputTag("tpToGlbTrackAssociation"),

    beamSpot = cms.InputTag("offlineBeamSpot"),

    useFabsEta = cms.bool(True),
    nint = cms.int32(25),
    min = cms.double(0),
    max = cms.double(2.5),

    nintPhi = cms.int32(25),
    minPhi = cms.double(-3.15),
    maxPhi = cms.double(3.15),

    nintpT = cms.int32(25),
    minpT = cms.double(0.0),
    maxpT = cms.double(3100.0),

    nintHit = cms.int32(75),
    minHit = cms.double(0.0),
    maxHit = cms.double(75.0),

    useInvPt = cms.bool(False)
)


