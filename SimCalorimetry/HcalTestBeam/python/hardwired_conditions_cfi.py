import FWCore.ParameterSet.Config as cms

# HCAL setup suitable for MC simulation and production (no ElectronicsMapping)
hcal_db_producer = cms.ESProducer("HcalDbProducer")

hcal_es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    fromDDD = cms.untracked.bool(False),
    toGet = cms.untracked.vstring('Pedestals', 
        'PedestalWidths', 
        'Gains', 
        'GainWidths', 
        'QIEShape', 
        'QIEData', 
        'ChannelQuality', 
        'ElectronicsMap')
)



