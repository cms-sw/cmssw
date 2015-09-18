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
  if (hasattr(process,"simMuonGEMDigis")) :
    if ( not hasattr(process.simMuonGEMDigis,"mixLabel")) :
      process.simMuonGEMDigis.mixLabel = cms.string("mix")
  if ( hasattr(process,"simMuonME0Digis")) :
    if ( not hasattr(process.simMuonME0Digis,"mixLabel")) :
      process.simMuonME0Digis.mixLabel = cms.string("mix")
  return process

def customise_Validation(process):
  process.load('Validation.MuonGEMHits.gemSimValid_cff')
  process.genvalid_all += process.gemSimValid
  return process

def customise_harvesting(process):
  process.load('Validation.MuonGEMHits.gemPostValidation_cff')
  process.genHarvesting += process.gemPostValidation
  process.load('DQMServices.Components.EDMtoMEConverter_cff')
  process.genHarvesting += process.EDMtoMEConverter
  return process

