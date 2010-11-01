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
    jetCorrectionLevel = cms.string("abs"),
    ## different ways to calculate a neutrino pz:
    ## -1 : take MET as neutrino directly, i.e. pz = 0
    ## or use mW = 80.4 GeV to solve the quadratic equation for the neutrino pz;
    ## if two real solutions...
    ##  0 : take the one closer to the lepton pz if neutrino pz < 300 GeV,
    ##      otherwise the more central one
    ##  1 : always take the one closer to the lepton pz
    ##  2 : always take the more central one, i.e. minimize neutrino pz
    ##  3 : maximize the cosine of the angle between lepton and reconstructed W
    ## in all these cases (0, 1, 2, 3), only the real part is used if solutions are complex
    neutrinoSolutionType = cms.int32(-1)
)
