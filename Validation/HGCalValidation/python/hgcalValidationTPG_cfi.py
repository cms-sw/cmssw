import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from L1Trigger.L1THGCal.egammaIdentification import egamma_identification_drnn_cone

hgcalTrigPrimValidation = DQMEDAnalyzer(
        "HGCalTriggerValidator",
        TriggerCells = cms.InputTag('hgcalConcentratorProducer:HGCalConcentratorProcessorSelection'),
        Clusters = cms.InputTag('hgcalBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering'),
        Multiclusters = cms.InputTag('hgcalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
        Towers = cms.InputTag('hgcalTowerProducer:HGCalTowerProcessor'),
        EGIdentification = egamma_identification_drnn_cone.clone()
       )

