import FWCore.ParameterSet.Config as cms

process.mix.hcal.doNoise = cms.bool(False)
process.mix.hcal.doEmpty = cms.bool(False)
process.mix.hcal.doHPDNoise = cms.bool(False)
process.mix.hcal.doIonFeedback = cms.bool(False)
process.mix.hcal.doThermalNoise = cms.bool(False)
process.mix.ecal.doNoise = cms.bool(False)
process.mix.pixel.AddNoise = cms.bool(False)
process.mix.strip.AddNoise = cms.bool(False)
