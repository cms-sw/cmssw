import FWCore.ParameterSet.Config as cms

def customise2019(process):
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'L1simulation_step'):
        process=customise_L1Emulator2019(process,'pt0')
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process)
    if hasattr(process,'HLTSchedule'):
        process=customise_gem_hlt(process)
    return process

def customise2023(process):
    process = customise2019(process)
    if hasattr(process,'L1simulation_step'):
        process=customise_L1Emulator2023(process,'pt0')
    return process

def customise_Digi(process):
    # original from SLHC:
    # ----------------------------
    # process.RandomNumberGeneratorService.simMuonGEMDigis = cms.PSet(
    #     initialSeed = cms.untracked.uint32(1234567),
    #     engineName = cms.untracked.string('HepJamesRandom')
    # )
    process.mix.mixObjects.mixSH.crossingFrames.append('MuonGEMHits')
    process.mix.mixObjects.mixSH.input.append(cms.InputTag("g4SimHits","MuonGEMHits"))
    process.mix.mixObjects.mixSH.subdets.append('MuonGEMHits')
    process.load('SimMuon.GEMDigitizer.muonGEMDigi_cff')
    process.muonDigi += process.muonGEMDigi
    # process.load('SimCalorimetry.Configuration.SimCalorimetry_cff')
    # process.digitisation_step.remove(process.calDigi)
    # process.load('SimCalorimetry.Configuration.ecalDigiSequence_cff')
    # process.load('SimCalorimetry.Configuration.hcalDigiSequence_cff')
    # process.load('SimCalorimetry.Configuration.castorDigiSequence_cff')
    # process.digitisation_step.remove(process.ecalDigiSequence)
    # process.digitisation_step.remove(process.hcalDigiSequence)
    # process.digitisation_step.remove(process.castorDigiSequence)
    process=outputCustoms(process)
    # ----------------------------

    # original from 75X:
    # ----------------------------
    # from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only
    # process = customize_digi_addGEM_muon_only(process)
    # process.simMuonGEMDigis.mixLabel = cms.string("mix")
    # process.simMuonRPCDigis.digiModel = cms.string('RPCSimAverageNoiseEff')
    # ----------------------------
    return process

#
# Here come the L1 Trigger Customizations, below comes from 620 SLHC
# ------------------------------------------------------------------
# def customise_L1Emulator2019(process, ptdphi):
#     from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigisPostLS2_cfi import cscTriggerPrimitiveDigisPostLS2
#     process.simCscTriggerPrimitiveDigis = cscTriggerPrimitiveDigisPostLS2
#     process.simCscTriggerPrimitiveDigis.clctSLHC.clctNplanesHitPattern = 3
#     process.simCscTriggerPrimitiveDigis.clctSLHC.clctPidThreshPretrig = 2
#     process.simCscTriggerPrimitiveDigis.clctParam07.clctPidThreshPretrig = 2
#     ## give a random number generator
#     process.RandomNumberGeneratorService.simCscTriggerPrimitiveDigis = cms.PSet(
#         initialSeed = cms.untracked.uint32(1234567),
#         engineName = cms.untracked.string('HepJamesRandom')
#     )
#     return process

# def customise_L1Emulator2023(process, ptdphi):
#     from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigisPostLS3_cfi import cscTriggerPrimitiveDigisPostLS3
#     process.simCscTriggerPrimitiveDigis = cscTriggerPrimitiveDigisPostLS3
#     process.simCscTriggerPrimitiveDigis.clctSLHC.clctNplanesHitPattern = 3
#     process.simCscTriggerPrimitiveDigis.clctSLHC.clctPidThreshPretrig = 2
#     process.simCscTriggerPrimitiveDigis.clctParam07.clctPidThreshPretrig = 2
#     ## ME21 has its own SLHC processors
#     process.simCscTriggerPrimitiveDigis.alctSLHCME21 = process.simCscTriggerPrimitiveDigis.alctSLHC.clone()
#     process.simCscTriggerPrimitiveDigis.clctSLHCME21 = process.simCscTriggerPrimitiveDigis.clctSLHC.clone()
#     process.simCscTriggerPrimitiveDigis.alctSLHCME21.alctNplanesHitPattern = 3
#     process.simCscTriggerPrimitiveDigis.alctSLHCME21.runME21ILT = cms.bool(True)
#     process.simCscTriggerPrimitiveDigis.clctSLHCME21.clctNplanesHitPattern = 3
#     process.simCscTriggerPrimitiveDigis.clctSLHCME21.clctPidThreshPretrig = 2
#     return process
# ------------------------------------------------------------------

def customise_DigiToRaw(process):
    return process

def customise_RawToDigi(process):
    return process

def customise_Reco(process):
    process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
    process.muonlocalreco += process.gemRecHits
    process.standAloneMuons.STATrajBuilderParameters.EnableGEMMeasurement = cms.bool(True)
    process.standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableGEMMeasurement = cms.bool(True)
    process.refittedStandAloneMuons.STATrajBuilderParameters.EnableGEMMeasurement = cms.bool(True)
    process.refittedStandAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableGEMMeasurement = cms.bool(True)
    process=outputCustoms(process)
    return process

def customise_DQM(process):
    return process

def customise_Validation(process):
    process.load('Validation.Configuration.gemSimValid_cff')
    process.load('Validation.MuonGEMDigis.MuonGEMDigis_cfi')
    process.genvalid_all += process.gemSimValid
    process.genvalid_all += process.gemDigiValidation
    # Additional in SLHC: 2B checked whether needed / necessary
    # ---------------------------------------------------------
    process.load('Validation.RecoMuon.MuonTrackValidator_cfi')
    process.load('SimMuon.MCTruth.MuonAssociatorByHits_cfi')
    process.muonAssociatorByHitsCommonParameters.useGEMs = cms.bool(True)
    process.muonTrackValidator.useGEMs = cms.bool(True)
    # ---------------------------------------------------------
    return process

def customise_harvesting(process):
    process.load('Validation.Configuration.gemPostValidation_cff')
    process.genHarvesting += process.gemPostValidation
    return process

def outputCustoms(process):
    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonGEMDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simMuonGEMCSCPadDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_gemRecHits_*_*')
    return process


def customise_gem_hlt(process):
    process.hltL2OfflineMuonSeeds.EnableGEMMeasurement = cms.bool( True )
    process.hltL2Muons.L2TrajBuilderParameters.EnableGEMMeasurement = cms.bool( True )
    process.hltL2Muons.BWFilterParameters.EnableGEMMeasurement = cms.bool( True )
    return process


