import FWCore.ParameterSet.Config as cms

# SVTagInfo adapter for VertexHistory 
svTagInfoVertexAdapter = cms.EDProducer("SVTagInfoVertexAdapter",
    svTagInfoProducer = cms.untracked.InputTag("secondaryVertexTagInfos")
)

