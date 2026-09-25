import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2

trackingParticleRecoTrackAssociationTable = cms.EDProducer(
    "TrackingParticleRecoTrackAssociationFlatTableProducer",
    src             = cms.InputTag("trackingParticleRecoTrackAsssociation"),
    keySrc          = cms.InputTag("mix", "MergedTrackTruth"),
    valSrc          = cms.InputTag("generalTracks"),
    name            = cms.string("TPAssoc"),
    doc             = cms.string("TrackingParticle -> reco::Track associations (dense per TP)"),
    linksName       = cms.string("TPAssocLinks"),
    linksDoc        = cms.string("Flattened TP -> reco::Track links: target index and association score"),
    scorePrecision  = cms.int32(14),
    skipNonExistingSrc = cms.bool(False),
)

recoTrackTrackingParticleAssociationTable = cms.EDProducer(
    "RecoTrackTrackingParticleAssociationFlatTableProducer",
    src             = cms.InputTag("trackingParticleRecoTrackAsssociation"),
    keySrc          = cms.InputTag("generalTracks"),
    valSrc          = cms.InputTag("mix", "MergedTrackTruth"),
    name            = cms.string("TrackAssoc"),
    doc             = cms.string("reco::Track -> TrackingParticle associations (dense per Track)"),
    linksName       = cms.string("TrackAssocLinks"),
    linksDoc        = cms.string("Flattened reco::Track -> TP links: target index and association score"),
    scorePrecision  = cms.int32(14),
    skipNonExistingSrc = cms.bool(False),
)

premix_stage2.toModify(trackingParticleRecoTrackAssociationTable, keySrc = "mixData:MergedTrackTruth")
premix_stage2.toModify(recoTrackTrackingParticleAssociationTable, valSrc = "mixData:MergedTrackTruth")

trackingParticleTrackAssociationTablesTask = cms.Task(
    trackingParticleRecoTrackAssociationTable,
    recoTrackTrackingParticleAssociationTable,
)
