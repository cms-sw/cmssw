import FWCore.ParameterSet.Config as cms

#
# module to make the mvaComputer
#
findTtFullHadSignalSelMVA = cms.EDProducer("TtFullHadSignalSelMVAComputer",
    ## jet input
    jets  = cms.InputTag("selectedPatJets")
)
# foo bar baz
# 236fHFwqeQa1F
