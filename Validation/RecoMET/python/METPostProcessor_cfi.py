import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

################# Postprocessing #########################
METPostprocessing = DQMEDHarvester('METTesterPostProcessor')  

################ Postprocessing Harvesting #########################
METPostprocessingHarvesting = DQMEDHarvester('METTesterPostProcessorHarvesting',
                    METTypeRECO = cms.InputTag("PfMetT1"),
                    METTypeMiniAOD = cms.InputTag("slimmedMETs")
                   )  
