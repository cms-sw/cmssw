import FWCore.ParameterSet.Config as cms

# TrackingParticle (MC truth) selectors
muonTPSet = cms.PSet(
    src = cms.InputTag("mix", "MergedTrackTruth"),
    pdgId = cms.vint32(13, -13),
    tip = cms.double(3.5),
    lip = cms.double(30.0),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    minRapidity = cms.double(-2.4),
    maxRapidity = cms.double(2.4),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    chargedOnly = cms.bool(True)
)

displacedMuonTPSet = cms.PSet(
    src = cms.InputTag("mix", "MergedTrackTruth"),
    pdgId = cms.vint32(13, -13),
    tip = cms.double(85.),  # radius to have at least the 3 outermost TOB layers
    lip = cms.double(210.), # z to have at least the 3 outermost TEC layers
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    minRapidity = cms.double(-2.4),
    maxRapidity = cms.double(2.4),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    chargedOnly = cms.bool(True)
)

#muonTP = cms.EDFilter("TrackingParticleSelector",
#    muonTPSet
#)

# RecoTrack selectors
#muonGlb = cms.EDFilter("RecoTrackSelector",
#    src = cms.InputTag("globalMuons"),
#    tip = cms.double(3.5),
#    lip = cms.double(30.0),
#    minHit = cms.int32(8),
#    maxChi2 = cms.double(999),
#    ptMin = cms.double(0.8),
#    quality = cms.string("Chi2"),
#    minRapidity = cms.double(-2.5),
#    maxRapidity = cms.double(2.5)
#)
#
#muonSta = cms.EDFilter("RecoTrackSelector",
#    src = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
#    tip = cms.double(999.0),
#    lip = cms.double(999.0),
#    minHit = cms.int32(1),
#    maxChi2 = cms.double(999),
#    ptMin = cms.double(0.8),
#    quality = cms.string("Chi2"),
#    minRapidity = cms.double(-2.5),
#    maxRapidity = cms.double(2.5)
#)

#muonSelector_step = cms.Sequence(muonTP+muonGlb+muonSta)

#muonSelector_seq = cms.Sequence(muonTP)
