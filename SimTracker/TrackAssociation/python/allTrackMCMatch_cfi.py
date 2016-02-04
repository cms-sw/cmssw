import FWCore.ParameterSet.Config as cms

allTrackMCMatch = cms.EDFilter("GenParticleMatchMerger",
    src = cms.VInputTag(cms.InputTag("trackMCMatch"), cms.InputTag("standAloneMuonsMCMatch"), cms.InputTag("globalMuonsMCMatch"))
)


