import FWCore.ParameterSet.Config as cms

topElsSkimValidation = cms.EDAnalyzer('TopSkimValidation',
                                   processName   = cms.string("HLT8E29"),
                                   triggerNames  = cms.untracked.vstring('HLT_Ele15_LW_L1R','HLT_Ele15_SC10_LW_L1R','HLT_Ele20_LW_L1R'),
                                   leptonPtcut   = cms.double(20.0),
                                   HLTPtcut      = cms.double(18.0),
                                   leptonFlavor  = cms.string("els"),
                                   leptonInputTag= cms.InputTag("gsfElectrons")
                                     )
