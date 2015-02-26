import FWCore.ParameterSet.Config as cms
def customise2023(process):
  if hasattr(process,'dqmHarvesting'):
    process=customise_harvesting(process)
  if hasattr(process,'validation_step'):
    process=customise_Validation(process)
  return process

def customise_Validation(process):
  process.load('Validation.MuonGEMHits.MuonGEMHits_cfi')
  process.genvalid_all += process.gemHitsValidation
  return process

def customise_harvesting(process):
  #process.load('Validation.Configuration.gemPostValidation_cff')
  #process.genHarvesting += process.gemPostValidation
  return process
