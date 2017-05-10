import FWCore.ParameterSet.Config as cms

################# Postprocessing #########################
METPostprocessing = cms.EDProducer('METTesterPostProcessor')  

################ Postprocessing Harvesting #########################
METPostprocessingHarvesting = cms.EDProducer('METTesterPostProcessorHarvesting',
                    METTypeRECO = cms.InputTag("PfMetT1"),
                    METTypeMiniAOD = cms.InputTag("slimmedMETs")
                   )  
