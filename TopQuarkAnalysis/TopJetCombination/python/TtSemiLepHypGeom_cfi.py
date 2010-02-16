import FWCore.ParameterSet.Config as cms

#
# module to make the geom hypothesis
#
ttSemiLepHypGeom = cms.EDProducer("TtSemiLepHypGeom",
    ## met input 
    mets  = cms.InputTag("patMETs"),
    ## jet input 
    jets  = cms.InputTag("selectedPatJets"),
    ## lepton input
    leps  = cms.InputTag("selectedPatMuons"),
    ## jet combination
    match = cms.InputTag("findTtSemiLepJetCombGeom"),
    ## specify jet correction level as
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
