# The following comments couldn't be translated into the new config version:

# services

import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalFullValid")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# geometry (Only Ecal)
process.load("Geometry.EcalCommonData.EcalOnly_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

# ECAL hits validation sequence
process.load("Validation.EcalHits.ecalSimHitsValidationSequence_cff")

# ECAL digis validation sequence
process.load("Validation.EcalDigis.ecalDigisValidationSequence_cff")

# ECAL rechits validation sequence
process.load("Validation.EcalRecHits.ecalRecHitsValidationSequence_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:hits.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.simhits = cms.Sequence(process.ecalSimHitsValidationSequence)
process.digis = cms.Sequence(process.ecalDigisValidationSequence)
process.rechits = cms.Sequence(process.ecalRecHitsValidationSequence)
process.p1 = cms.Path(process.simhits*process.digis*process.rechits)
process.DQM.collectorHost = ''

