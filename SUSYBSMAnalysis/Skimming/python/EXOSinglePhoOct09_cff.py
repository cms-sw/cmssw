import FWCore.ParameterSet.Config as cms

singlePhotonMod = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>50"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
)

exoticaSinglePhoHighetSeq=cms.Sequence(singlePhotonMod)
