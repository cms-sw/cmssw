import FWCore.ParameterSet.Config as cms
def customise2023(process):
 # if hasattr(process,'digitisation_step') :
 #   process=customise_digitization(process)
  if hasattr(process,'dqmHarvesting'):
    process=customise_harvesting(process)
  if hasattr(process,'validation_step'):
    process=customise_Validation(process)
  return process
def customise_digitization(process):
  from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only
  process = customize_digi_addGEM_muon_only(process)
  process.simMuonGEMDigis.mixLabel = cms.string("mix")
  #process.simMuonRPCDigis.digiModel = cms.string('RPCSimParam')
  #process.simMuonME0PseudoDigis.mixLabel = cms.string("mix")
  process.digitisation_step.remove(process.simMuonRPCDigis)
  return process

def customise_Validation(process):
  #process.load('Validation.MuonGEMHits.MuonGEMHits_cfi')
  process.load('Validation.MuonME0Validation.me0SimValid_cff')
  process.genvalid_all += process.me0SimValid
  return process

def customise_harvesting(process):
  #process.load('Validation.MuonGEMHits.MuonGEMHits_cfi')
  #process.load('Validation.MuonME0Hits.me0PostValidation_cff')
  #process.genHarvesting += process.me0PostValidation
  process.load('DQMServices.Components.EDMtoMEConverter_cff')
  process.genHarvesting += process.EDMtoMEConverter
  return process


