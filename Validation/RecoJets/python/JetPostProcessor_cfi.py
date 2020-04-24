import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

################# Postprocessing #########################
JetPostprocessing = DQMEDHarvester('JetTesterPostProcessor',
                    JetTypeRECO = cms.InputTag("ak4PFJetsCHS"),
                    JetTypeMiniAOD = cms.InputTag("slimmedJets")
                   )  
