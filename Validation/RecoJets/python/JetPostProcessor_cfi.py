import FWCore.ParameterSet.Config as cms

################# Postprocessing #########################
JetPostprocessing = cms.EDAnalyzer('JetTesterPostProcessor',
                    JetTypeRECO = cms.InputTag("ak4PFJetsCHS"),
                    JetTypeMiniAOD = cms.InputTag("slimmedJets")
                   )  
