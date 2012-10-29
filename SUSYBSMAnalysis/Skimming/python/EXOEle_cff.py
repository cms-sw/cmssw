import FWCore.ParameterSet.Config as cms


########################################
# definitions for cut on HLT variables #
########################################

lowetMod = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>30"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
)

medetMod = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>50"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
)

highetMod = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>80"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
)

exoticaEleLowetSeq=cms.Sequence(lowetMod)
exoticaEleMedetSeq=cms.Sequence(medetMod)
exoticaEleHighetSeq=cms.Sequence(highetMod)

############################################
# defintions for cuts on RECO variables    #
############################################
# low and medium Et paths currently identical
# low Et path is currently maintainted for
# easy addition of isolation requirements in the future

lowetModR  = cms.EDFilter("HLTHighLevel",
                                                   TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29"),
                                                   HLTPaths = cms.vstring('HLT_Ele15_LW_L1R',
                                                                                                                          'HLT_Photon20_L1R'),
                                                   andOr = cms.bool(True),
                                                   eventSetupPathsKey = cms.string(''),
                                                   throw = cms.bool(True)
                                                   )
lowetsel = cms.EDFilter("GsfElectronSelector",
                                                src =cms.InputTag("gsfElectrons"),
                                                cut =cms.string("superCluster().get().energy()*sin(theta())>30")
                                                )

lowfilter = cms.EDFilter("CandViewCountFilter",
                                                  src = cms.InputTag("lowetsel"),
                                                  minNumber = cms.uint32(1),
                                                  )

medetModR  = cms.EDFilter("HLTHighLevel",
                                                   TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29"),
                                                   HLTPaths = cms.vstring("HLT_Ele15_LW_L1R",
                                                                                                                          "HLT_Photon20_L1R"),
                                                   andOr = cms.bool(True),
                                                   eventSetupPathsKey = cms.string(''),
                                                   throw = cms.bool(True)
                                                   )
medetsel = cms.EDFilter("GsfElectronSelector",
                                                src =cms.InputTag("gsfElectrons"),
                                                cut =cms.string("superCluster().get().energy()*sin(theta())>50")
                                                )

medfilter = cms.EDFilter("CandViewCountFilter",
                                                  src = cms.InputTag("medetsel"),
                                                  minNumber = cms.uint32(1),
                                                  )

exoticaEleLowetSeqReco=cms.Sequence(lowetModR * lowetsel * lowfilter)
exoticaEleMedetSeqReco=cms.Sequence(medetModR * medetsel * medfilter)
exoticaEleHighetSeqReco=cms.Sequence(highetMod)
