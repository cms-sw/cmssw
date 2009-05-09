import FWCore.ParameterSet.Config as cms

#
# module to make the wMassMaxSumPt hypothesis
#
ttSemiLepHypWMassMaxSumPt = cms.EDProducer("TtSemiLepHypWMassMaxSumPt",
    ## met input
    mets  = cms.InputTag("layer1METs"),
    ## jet input
    jets  = cms.InputTag("selectedLayer1Jets"),
    ## lepton input
    leps  = cms.InputTag("selectedLayer1Muons"),
    ## maximal number of jets to be considered
    maxNJets = cms.int32(4),
    ## nominal WMass parameter
    wMass    = cms.double(80.413)
)


