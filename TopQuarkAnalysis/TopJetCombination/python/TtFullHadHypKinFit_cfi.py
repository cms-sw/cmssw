
import FWCore.ParameterSet.Config as cms

#
# module to make the kinematic fit hypothesis
#
ttFullHadHypKinFit = cms.EDProducer("TtFullHadHypKinFit",
    ## jet input
    jets  = cms.InputTag("selectedLayer1Jets"),
    ## kin fit results
    match        = cms.InputTag("kinFitTtFullHadEventHypothesis"),
    status       = cms.InputTag("kinFitTtFullHadEventHypothesis","Status"),
    lightQTag    = cms.InputTag("kinFitTtFullHadEventHypothesis","PartonsLightQ"),
    lightQBarTag = cms.InputTag("kinFitTtFullHadEventHypothesis","PartonsLightQBar"),
    lightPTag    = cms.InputTag("kinFitTtFullHadEventHypothesis","PartonsLightP"),
    lightPBarTag = cms.InputTag("kinFitTtFullHadEventHypothesis","PartonsLightPBar"),
    bTag         = cms.InputTag("kinFitTtFullHadEventHypothesis","PartonsB"),
    bBarTag      = cms.InputTag("kinFitTtFullHadEventHypothesis","PartonsBBar")
)
