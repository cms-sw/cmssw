import FWCore.ParameterSet.Config as cms

################# Postprocessing #########################
METPostprocessing = cms.EDAnalyzer('METTesterPostProcessor')  

################ Postprocessing Harvesting #########################
METPostprocessingHarvesting = cms.EDAnalyzer('METTesterPostProcessorHarvesting',
                    METTypeRECO = cms.InputTag("PfMetT1"),
                    METTypeMiniAOD = cms.InputTag("slimmedMETs")
                   )  
