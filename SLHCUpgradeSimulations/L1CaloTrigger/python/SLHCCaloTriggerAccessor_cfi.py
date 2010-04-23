import FWCore.ParameterSet.Config as cms

CaloTriggerAccessor = cms.EDFilter("SLHCCaloTriggerAccessor",
    L1EGamma       = cms.InputTag("L1ExtraMaker","EGamma"),
    L1IsoEGamma    = cms.InputTag("L1ExtraMaker","IsoEGamma"),
    L1Tau          = cms.InputTag("L1ExtraMaker","Taus"),
    L1IsoTau       = cms.InputTag("L1ExtraMaker","IsoTaus"),
    Jets           = cms.InputTag("L1ExtraMaker","Jets"),
    OutputFileName = cms.string('analysis.root')
)

