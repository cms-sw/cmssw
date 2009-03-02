import FWCore.ParameterSet.Config as cms

# SVTagInfo proxy for VertexHistory 
svTagInfoVertexProxy = cms.EDProducer("SVTagInfoVertexProxy",
    svTagInfoProducer = cms.untracked.InputTag("secondaryVertexTagInfos")
)

