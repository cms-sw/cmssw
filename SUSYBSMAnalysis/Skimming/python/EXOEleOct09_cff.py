import FWCore.ParameterSet.Config as cms

lowetMod = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter","","HLT"),      # filter or collection
    cut     = cms.string("pt>30"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
)

medetMod = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter","","HLT"),      # filter or collection
    cut     = cms.string("pt>50"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
)

highetMod = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter","","HLT"),      # filter or collection
    cut     = cms.string("pt>80"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
)

exoticaEleLowetSeq=cms.Sequence(lowetMod)
exoticaEleMedetSeq=cms.Sequence(medetMod)
exoticaEleHighetSeq=cms.Sequence(highetMod)
