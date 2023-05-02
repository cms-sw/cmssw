import FWCore.ParameterSet.Config as cms

allTrackMCMatch = cms.EDProducer("GenParticleMatchMerger",
                                 src = cms.VInputTag(cms.InputTag("trackMCMatch"),
                                                     cms.InputTag("standAloneMuonsMCMatch"),
                                                     cms.InputTag("globalMuonsMCMatch")))


