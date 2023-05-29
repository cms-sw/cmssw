import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from L1Trigger.L1THGCal.egammaIdentification import egamma_identification_drnn_cone

hgcalTrigPrimValidation = DQMEDAnalyzer(
        "HGCalTriggerValidator",
        TriggerCells = cms.InputTag('l1tHGCalConcentratorProducer:HGCalConcentratorProcessorSelection'),
        Clusters = cms.InputTag('l1tHGCalBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering'),
        Multiclusters = cms.InputTag('l1tHGCalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
        Towers = cms.InputTag('l1tHGCalTowerProducer:HGCalTowerProcessor'),
        EGIdentification = egamma_identification_drnn_cone.clone()
       )

