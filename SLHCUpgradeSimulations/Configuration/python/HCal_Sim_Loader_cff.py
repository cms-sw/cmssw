import FWCore.ParameterSet.Config as cms

def customise(process):
     # use hardcoded values
     process.es_hardcode.toGet.extend(['Gains', 'Pedestals', 'PedestalWidths', 'QIEData', 'ElectronicsMap','ChannelQuality','RespCorrs','ZSThresholds','LutMetadata','L1TriggerObjects','TimeCorrs','PFCorrs','LUTCorrs'])
     process.es_hardcode.H2Mode = cms.untracked.bool(False)
     process.es_hardcode.SLHCMode = cms.untracked.bool(True)
     process.es_prefer_hcalHardcode = cms.ESPrefer("HcalHardcodeCalibrations", "es_hardcode")
     process.output.outputCommands.extend([ 'keep *_towerMakerWithHO_*_*' ])
     process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EMV'
     process.g4SimHits.HCalSD.TestNumberingScheme = True
     process.HcalTopologyIdealEP.SLHCMode = cms.untracked.bool(True)
     return (process)
