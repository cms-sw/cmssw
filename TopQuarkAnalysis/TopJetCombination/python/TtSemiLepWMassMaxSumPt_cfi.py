import FWCore.ParameterSet.Config as cms

#
# module to make the wMassMaxSumPt hypothesis
#
ttSemiLepWMassMaxSumPt = cms.EDProducer("TtSemiLepWMassMaxSumPt",
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    jets  = cms.InputTag("selectedLayer1Jets"),
    maxNJets = cms.uint32(4),
    wMass    = cms.double(80.413)
)


