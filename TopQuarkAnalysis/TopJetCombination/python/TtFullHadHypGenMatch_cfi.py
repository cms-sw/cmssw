import FWCore.ParameterSet.Config as cms

#
# module to make the genMatch hypothesis
#
ttFullHadHypGenMatch = cms.EDProducer("TtFullHadHypGenMatch",
    ## jet input
    jets  = cms.InputTag("selectedPatJets"),
    ## gen match hypothesis input
    match = cms.InputTag("ttFullHadJetPartonMatch"),
    ## specify jet correction level as, Uncorrected, L1Offset, L2Relative, L3Absolute, L4Emf,
    ## L5Hadron, L6UE, L7Parton, a flavor specification will be added automatically, when chosen                                      
    jetCorrectionLevel = cms.string("L3Absolute")   
)


