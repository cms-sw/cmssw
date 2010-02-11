import FWCore.ParameterSet.Config as cms

singlePhotonMod = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter","","HLT"),      # filter or collection
    cut     = cms.string("pt>15"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
)

exoticaSinglePhoHighetSeq=cms.Sequence(singlePhotonMod)
