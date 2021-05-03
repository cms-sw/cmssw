import FWCore.ParameterSet.Config as cms

# TrackingParticle skims used in muon validation

import SimMuon.MCTruth.trackingParticleMuon_cfi
TPrefs = SimMuon.MCTruth.trackingParticleMuon_cfi.trackingParticleMuon.clone()

TPmu = TPrefs.clone()
TPmu_seq = cms.Sequence( TPmu )

TPpfmu = TPrefs.clone()
TPpfmu.skim = 'pf'
TPpfmu_seq = cms.Sequence( TPpfmu )

TPtrack = TPrefs.clone()
TPtrack.skim = 'track'
TPtrack.ptmin = 2.
TPtrack_seq = cms.Sequence( TPtrack )

# TrackingParticle selectors for signal efficiencies

muonTPSet = cms.PSet(
    src = cms.InputTag("mix", "MergedTrackTruth"),
    pdgId = cms.vint32(13, -13),
    tip = cms.double(3.5),
    lip = cms.double(30.0),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    ptMax = cms.double(1e100),
    minRapidity = cms.double(-2.4),
    maxRapidity = cms.double(2.4),
    signalOnly = cms.bool(True),
    intimeOnly = cms.bool(True),  # discard OOT PU
    stableOnly = cms.bool(True),  # discard decays in flight from the signal event
    chargedOnly = cms.bool(True)
)

me0MuonTPSet = cms.PSet(
    src = cms.InputTag("mix", "MergedTrackTruth"),
    pdgId = cms.vint32(13, -13),
    tip = cms.double(3.5),
    lip = cms.double(30.0),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    ptMax = cms.double(1e100),
    minRapidity = cms.double(-2.8),
    maxRapidity = cms.double(2.8),
    signalOnly = cms.bool(True),
    intimeOnly = cms.bool(True),  # discard OOT PU
    stableOnly = cms.bool(True),  # discard decays in flight from the signal event
    chargedOnly = cms.bool(True)
)

displacedMuonTPSet = cms.PSet(
    src = cms.InputTag("mix", "MergedTrackTruth"),
    pdgId = cms.vint32(13, -13),
    tip = cms.double(85.),  # radius to have at least the 3 outermost TOB layers
    lip = cms.double(210.), # z to have at least the 3 outermost TEC layers
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    ptMax = cms.double(1e100),
    minRapidity = cms.double(-2.4),
    maxRapidity = cms.double(2.4),
    signalOnly = cms.bool(True),
    intimeOnly = cms.bool(True),  # discard OOT PU
    stableOnly = cms.bool(True), # accept only TP from the Generator (linked to GenParticles)
    chargedOnly = cms.bool(True)
)

cosmicMuonTPSet = cms.PSet(
    src = cms.InputTag("mix", "MergedTrackTruth"),
    pdgId = cms.vint32(13, -13),
    tip = cms.double(85.),  # radius to have at least the 3 outermost TOB layers
    lip = cms.double(210.), # z to have at least the 3 outermost TEC layers
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    ptMax = cms.double(1e100),
    minRapidity = cms.double(-2.4),
    maxRapidity = cms.double(2.4),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(True), # accept only TP from the Generator (linked to GenParticles)
    chargedOnly = cms.bool(True)
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(muonTPSet, src = "mixData:MergedTrackTruth")
premix_stage2.toModify(cosmicMuonTPSet, src = "mixData:MergedTrackTruth")
premix_stage2.toModify(displacedMuonTPSet, src = "mixData:MergedTrackTruth")
premix_stage2.toModify(me0MuonTPSet, src = "mixData:MergedTrackTruth")

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
