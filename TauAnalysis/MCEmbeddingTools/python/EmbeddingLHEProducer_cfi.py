import FWCore.ParameterSet.Config as cms


externalLHEProducer = cms.EDProducer("EmbeddingLHEProducer",
    src = cms.InputTag("selectedMuonsForEmbedding","",""),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices","","SELECT"),
    particleToEmbed = cms.int32(15),
    rotate180 = cms.bool(False),
    mirror = cms.bool(False),
    InitialRecoCorrection = cms.bool(True),
    studyFSRmode = cms.untracked.string("reco")
)

makeexternalLHEProducer = cms.Sequence( externalLHEProducer)
