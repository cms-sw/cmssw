import FWCore.ParameterSet.Config as cms

trackingParticleRecoTrackAsssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.InputTag('quickTrackAssociatorByHits'),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    label_tr = cms.InputTag("generalTracks"),
    ignoremissingtrackcollection=cms.untracked.bool(False)
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(trackingParticleRecoTrackAsssociation, label_tp = "mixData:MergedTrackTruth")

