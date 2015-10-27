import FWCore.ParameterSet.Config as cms
def customise2023(process):
  if hasattr(process,'digitisation_step') :
    process=customise_digitization(process)
  if hasattr(process, 'L1simulation_step') :
    process=customise_L1simulation(process)
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

def customise_L1simulation(process):
  if (not hasattr(process, 'caloConfigSource')) :
    process.load('L1Trigger.L1TCalorimeter.caloConfigStage1PP_cfi')
  from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import cscTriggerPrimitiveDigis
  process.simCscTriggerPrimitiveDigis = cscTriggerPrimitiveDigis
  process.simCscTriggerPrimitiveDigis.commonParam.isSLHC = True
  process.simCscTriggerPrimitiveDigis.commonParam.smartME1aME1b = True

  from Validation.MuonGEMDigis.MuonGEMDigis_cff import me11tmbSLHCGEM
  process.simCscTriggerPrimitiveDigis.commonParam.runME11ILT = cms.bool(True)
  process.simCscTriggerPrimitiveDigis.me11tmbSLHCGEM = me11tmbSLHCGEM
  process.simCscTriggerPrimitiveDigis.clctSLHC.clctNplanesHitPattern = 3
  process.simCscTriggerPrimitiveDigis.clctSLHC.clctPidThreshPretrig = 2
  process.simCscTriggerPrimitiveDigis.clctParam07.clctPidThreshPretrig = 2
  process.simCscTriggerPrimitiveDigis.GEMPadDigiProducer = "simMuonGEMPadDigis"

  from Validation.MuonGEMDigis.MuonGEMDigis_cff import me21tmbSLHCGEM
  process.simCscTriggerPrimitiveDigis.commonParam.runME21ILT = cms.bool(True)
  process.simCscTriggerPrimitiveDigis.me21tmbSLHCGEM = me21tmbSLHCGEM
  ## ME21 has its own SLHC processors
  process.simCscTriggerPrimitiveDigis.alctSLHCME21 = process.simCscTriggerPrimitiveDigis.alctSLHC.clone()
  process.simCscTriggerPrimitiveDigis.clctSLHCME21 = process.simCscTriggerPrimitiveDigis.clctSLHC.clone()
  process.simCscTriggerPrimitiveDigis.alctSLHCME21.alctNplanesHitPattern = 3
  #process.simCscTriggerPrimitiveDigis.alctSLHCME21.runME21ILT = cms.bool(True)
  process.simCscTriggerPrimitiveDigis.clctSLHCME21.clctNplanesHitPattern = 3
  process.simCscTriggerPrimitiveDigis.clctSLHCME21.clctPidThreshPretrig = 2
  return process

def customise_Validation(process):
  process.load('Validation.MuonGEMHits.gemSimValid_cff')
  process.genvalid_all += process.gemSimValid
  if ( hasattr(process,"me0SimValid") ) :
    process.genvalid_all += process.me0SimValid
  return process

def customise_harvesting(process):
  process.load('Validation.MuonGEMHits.gemPostValidation_cff')
  process.genHarvesting += process.gemPostValidation
  process.load('DQMServices.Components.EDMtoMEConverter_cff')
  process.genHarvesting += process.EDMtoMEConverter
  return process

