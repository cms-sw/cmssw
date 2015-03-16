import FWCore.ParameterSet.Config as cms

################# Postprocessing #########################
METPostprocessing = cms.EDAnalyzer('METTesterPostProcessor',
                    METTypeRECO = cms.InputTag("pfMetT1"),
                    METTypeMiniAOD = cms.InputTag("slimmedMETs")
                   )  
