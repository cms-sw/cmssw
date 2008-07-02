import FWCore.ParameterSet.Config as cms

#
# module to make the maxSumPtWMAss hypothesis
#
ttSemiHypothesisMaxSumPtWMass = cms.EDProducer("TtSemiHypothesisMaxSumPtWMass",
    leps  = cms.InputTag("selectedLayer1Muons")
    mets  = cms.InputTag("selectedLayer1METs"),
    jets  = cms.InputTag("selectedLayer1Jets"),
    match = cms.InputTag( ),
    nJetsMax = cms.uint32(4),
)


