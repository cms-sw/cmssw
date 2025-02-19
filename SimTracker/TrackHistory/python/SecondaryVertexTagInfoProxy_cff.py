import FWCore.ParameterSet.Config as cms

# SVTagInfo proxy for VertexHistory 
svTagInfoProxy = cms.EDProducer("SecondaryVertexTagInfoProxy",
    svTagInfoProducer = cms.untracked.InputTag("secondaryVertexTagInfos")
)

