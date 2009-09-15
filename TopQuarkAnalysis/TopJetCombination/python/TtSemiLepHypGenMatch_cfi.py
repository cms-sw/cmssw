import FWCore.ParameterSet.Config as cms

#
# module to make the genMatch hypothesis
#
ttSemiLepHypGenMatch = cms.EDProducer("TtSemiLepHypGenMatch",
    ## met input
    mets  = cms.InputTag("layer1METs"),
    ## jet input
    jets  = cms.InputTag("selectedLayer1Jets"),
    ## lepton input
    leps  = cms.InputTag("selectedLayer1Muons"),
    ## gen match hypothesis input
    match = cms.InputTag("ttSemiLepJetPartonMatch"),
    ## specify jet correction level as,
    ## No Correction : raw                                     
    ## L1Offset      : off
    ## L2Relative    : rel
    ## L3Absolute    : abs
    ## L4Emf         : emf
    ## L5Hadron      : had
    ## L6UE          : ue
    ## L7Parton      : part
    ## a flavor specification will be
    ## added automatically, when chosen
    jetCorrectionLevel = cms.string("abs")                                      
)


