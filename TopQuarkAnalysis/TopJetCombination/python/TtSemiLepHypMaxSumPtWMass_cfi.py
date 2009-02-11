import FWCore.ParameterSet.Config as cms

#
# module to make the maxSumPtWMAss hypothesis
#
ttSemiLepHypMaxSumPtWMass = cms.EDProducer("TtSemiLepHypMaxSumPtWMass",
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    jets  = cms.InputTag("selectedLayer1Jets"),
    maxNJets = cms.int32(4),
    wMass = cms.double(80.413)
)


