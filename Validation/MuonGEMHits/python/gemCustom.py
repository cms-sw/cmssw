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
  #process.simMuonME0Digis.mixLabel = cms.string("mix")
  #process.digitisation_step.remove(process.simMuonRPCDigis)
  return process

def customise_Validation(process):
  process.load('Validation.MuonGEMHits.MuonGEMHits_cfi')
  process.load('Validation.MuonGEMDigis.MuonGEMDigis_cfi')
  process.genvalid_all += process.gemHitsValidation
  process.genvalid_all += process.gemDigiValidation
  return process

def customise_harvesting(process):
  #process.load('Validation.Configuration.gemPostValidation_cff')
  #process.genHarvesting += process.gemPostValidation
  return process

