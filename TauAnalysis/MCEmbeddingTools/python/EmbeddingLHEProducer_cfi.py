import FWCore.ParameterSet.Config as cms


externalLHEProducer = cms.EDProducer("EmbeddingLHEProducer",
    src = cms.InputTag("selectedMuonsForEmbedding","",""),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices","","SELECT"),
    switchToMuonEmbedding = cms.bool(False),
    rotate180 = cms.bool(False),
    mirror = cms.bool(False),
    studyFSRmode = cms.untracked.string("reco")
)

makeexternalLHEProducer = cms.Sequence( externalLHEProducer)