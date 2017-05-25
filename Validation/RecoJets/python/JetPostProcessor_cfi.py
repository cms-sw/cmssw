import FWCore.ParameterSet.Config as cms

################# Postprocessing #########################
JetPostprocessing = cms.EDProducer('JetTesterPostProcessor',
                    JetTypeRECO = cms.InputTag("ak4PFJetsCHS"),
                    JetTypeMiniAOD = cms.InputTag("slimmedJets")
                   )  
