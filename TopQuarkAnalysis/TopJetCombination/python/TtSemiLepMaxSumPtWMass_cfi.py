import FWCore.ParameterSet.Config as cms

#
# module to make the maxSumPtWMAss hypothesis
#
ttSemiLepMaxSumPtWMass = cms.EDProducer("TtSemiLepMaxSumPtWMass",
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    jets  = cms.InputTag("selectedLayer1Jets"),
    nJetsMax = cms.uint32(4),
    wMass = cms.double(80.413)
)


