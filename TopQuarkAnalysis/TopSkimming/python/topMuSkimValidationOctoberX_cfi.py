import FWCore.ParameterSet.Config as cms

topMuSkimValidation = cms.EDAnalyzer('TopSkimValidation',
                                   processName   = cms.string("HLT8E29"),
                                   triggerNames  = cms.untracked.vstring("HLT_Mu9"),
                                   leptonPtcut   = cms.double(20.0),
                                   HLTPtcut      = cms.double(18.0),
                                   leptonFlavor  = cms.string("mus"),
                                   leptonInputTag= cms.InputTag("muons")
                                     )
