import FWCore.ParameterSet.Config as cms
def customise2023(process):
  if hasattr(process,'digitisation_step') :
    process=customise_digitization(process)
  if hasattr(process,'dqmHarvesting'):
    process=customise_harvesting(process)
  if hasattr(process,'validation_step'):
    process=customise_Validation(process)
  return process
def customise_digitization(process):
  from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only
  process = customize_digi_addGEM_muon_only(process)
  process.simMuonGEMDigis.mixLabel = cms.string("mix")
  # process.simMuonME0Digis.mixLabel = cms.string("mix")
  # process.digitisation_step.remove(process.simMuonRPCDigis)
  # process.simMuonRPCDigis.digiModel = cms.string('RPCSimParam')
  process.simMuonRPCDigis.digiModel = cms.string('RPCSimAverageNoiseEff')
  return process

def customise_Validation(process):
  #process.load('Validation.MuonGEMHits.MuonGEMHits_cfi')
  process.load('Validation.MuonGEMHits.gemSimValid_cff')
  process.load('Validation.MuonGEMDigis.MuonGEMDigis_cfi')
  process.genvalid_all += process.gemSimValid
  process.genvalid_all += process.gemDigiValidation
  return process

def customise_harvesting(process):
  #process.load('Validation.MuonGEMHits.MuonGEMHits_cfi')
  process.load('Validation.MuonGEMHits.gemPostValidation_cff')
  process.genHarvesting += process.gemPostValidation
  return process

