import FWCore.ParameterSet.Config as cms

#
# module to make the mvaComputer
#
findTtSemiLepSignalSelMVA = cms.EDProducer("TtSemiLepSignalSelMVAComputer",
    ## met input
    mets  = cms.InputTag("layer1METs"),
    ## jet input
    jets  = cms.InputTag("selectedLayer1Jets"),
    ## lepton input
    muons = cms.InputTag("selectedLayer1Muons"),
    elecs = cms.InputTag("selectedLayer1Electrons")
)
