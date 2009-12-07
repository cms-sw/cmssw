import FWCore.ParameterSet.Config as cms

#
# module to make the genMatch hypothesis
#
ttFullHadHypGenMatch = cms.EDProducer("TtFullHadHypGenMatch",
    ## jet input
    jets  = cms.InputTag("selectedLayer1Jets"),
    ## gen match hypothesis input
    match = cms.InputTag("ttFullHadJetPartonMatch"),
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


