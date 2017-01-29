import FWCore.ParameterSet.Config as cms

def customise_HcalPhase0(process):
    process.load("CalibCalorimetry/HcalPlugins/Hcal_Conditions_forGlobalTag_cff")

    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):
        process.mix.digitizers.hcal.TestNumbering=True

    process.es_hardcode.HEreCalibCutoff = cms.double(20.) #for aging

    process.es_hardcode.toGet = cms.untracked.vstring(
        'GainWidths',
        'RespCorrs'
        )


    if hasattr(process,'g4SimHits'):
        process=customise_Sim(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process)

    return process

#common stuff
def load_HcalHardcode(process):
    process.load("CalibCalorimetry/HcalPlugins/Hcal_Conditions_forGlobalTag_cff")
    process.es_hardcode.toGet = cms.untracked.vstring(
                'GainWidths',
                'MCParams',
                'RecoParams',
                'RespCorrs',
                'QIEData',
                'QIETypes',
                'Gains',
                'Pedestals',
                'PedestalWidths',
                'ChannelQuality',
                'ZSThresholds',
                'TimeCorrs',
                'LUTCorrs',
                'LutMetadata',
                'L1TriggerObjects',
                'PFCorrs',
                'ElectronicsMap',
                'FrontEndMap',
                'CovarianceMatrices',
                'SiPMParameters',
                'SiPMCharacteristics',
                'TPChannelParameters',
                'TPParameters',
                'FlagHFDigiTimeParams',
                )

    # Special Upgrade trick (if absent - regular case assumed)
    process.es_hardcode.GainWidthsForTrigPrims = cms.bool(True)
                
    return process

#intermediate customization (HF 2016 upgrades)
def customise_Hcal2016(process):
    process=load_HcalHardcode(process)
    
    #for now, use HE run1 conditions - SiPM/QIE11 not ready
    process.es_hardcode.testHFQIE10 = cms.bool(True)
    
    # to get reco to run
    if hasattr(process,'reconstruction_step'):
        process.hbheprereco.setNoiseFlags = cms.bool(False)
    
    return process
    
#intermediate customization (HCAL 2017, HE and HF upgrades - no SiPMs or QIE11)
def customise_Hcal2017(process):
    process=load_HcalHardcode(process)
    
    #for now, use HE run1 conditions - SiPM/QIE11 not ready
    process.es_hardcode.useHFUpgrade = cms.bool(True)
    
    # to get reco to run
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction_step'):
        process.hbheprereco.digiLabelQIE8 = cms.InputTag("simHcalDigis")
        process.hbheprereco.digiLabelQIE11 = cms.InputTag("simHcalDigis","HBHEQIE11DigiCollection")
        # process.hbheprereco.puCorrMethod = cms.int32(0)
        process.horeco.digiLabel = cms.InputTag("simHcalDigis")
        process.zdcreco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")
        process.zdcreco.digiLabelhcal = cms.InputTag("simHcalUnsuppressedDigis")
        process.hcalnoise.digiCollName = cms.string('simHcalDigis')
        process.hfprereco.digiLabel = cms.InputTag("simHcalDigis", "HFQIE10DigiCollection")
    if hasattr(process,'datamixing_step'):
        process=customise_mixing(process)
    if hasattr(process,'simHcalTriggerPrimitiveDigis'):
        process.simHcalTriggerPrimitiveDigis.upgradeHF = cms.bool(True)
    if hasattr(process,'dqmoffline_step'):
        process.digiTask.tagHBHE = cms.untracked.InputTag("simHcalDigis")
        process.digiTask.tagHF = cms.untracked.InputTag("simHcalDigis")
        process.digiTask.tagHO = cms.untracked.InputTag("simHcalDigis")

        #add phase1 digi task
        process.load('DQM.HcalTasks.DigiPhase1Task')
        process.dqmoffline_step += process.digiPhase1Task
        process.digiPhase1Task.tagHBHE = cms.untracked.InputTag("simHcalDigis","HBHEQIE11DigiCollection")
        process.digiPhase1Task.tagHO = cms.untracked.InputTag("simHcalDigis")
        process.digiPhase1Task.tagHF = cms.untracked.InputTag("simHcalDigis","HFQIE10DigiCollection")
        
    if hasattr(process,'validation_step'):
        process.AllHcalDigisValidation.digiLabel = cms.string("simHcalDigis")

    return process
    
#intermediate customization (HCAL 2017, HE and HF upgrades - w/ SiPMs & QIE11)
def customise_Hcal2017Full(process):
    process=customise_Hcal2017(process)
    
    #use HE phase1 conditions - test SiPM/QIE11
    process.es_hardcode.useHEUpgrade = cms.bool(True)

    if hasattr(process,'reconstruction_step'):
        # Customise HB/HE reco
        from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco
        process.globalReplace("hbheprereco", hbheprereco)
        process.hbheprereco.saveInfos = cms.bool(True)
        process.hbheprereco.digiLabelQIE8 = cms.InputTag("simHcalDigis")
        process.hbheprereco.digiLabelQIE11 = cms.InputTag("simHcalDigis", "HBHEQIE11DigiCollection")

    if hasattr(process,'simHcalTriggerPrimitiveDigis'):
        process.simHcalTriggerPrimitiveDigis.upgradeHE = cms.bool(True)

    return process
    
def customise_HcalPhase1(process):
    process=load_HcalHardcode(process)

    process.es_hardcode.HEreCalibCutoff = cms.double(100.) #for aging
    process.es_hardcode.useHBUpgrade = cms.bool(True)
    process.es_hardcode.useHEUpgrade = cms.bool(True)
    process.es_hardcode.useHFUpgrade = cms.bool(True)

    if hasattr(process,'g4SimHits'):
        process=customise_Sim(process)
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'reconstruction_step'):
        process=customise_Reco(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process)
    if hasattr(process,'simHcalTriggerPrimitiveDigis'):
        process.simHcalTriggerPrimitiveDigis.upgradeHF = cms.bool(True)
        process.simHcalTriggerPrimitiveDigis.upgradeHE = cms.bool(True)
        process.simHcalTriggerPrimitiveDigis.upgradeHB = cms.bool(True)
    process=customise_condOverRides(process)
    return process


def customise_Sim(process):
    process.g4SimHits.HCalSD.TestNumberingScheme = True

    return process

def customise_DigiToRaw(process):
    process.digi2raw_step.remove(process.hcalRawData)

    return process

def customise_RawToDigi(process):
    process.raw2digi_step.remove(process.hcalDigis)

    return process

def customise_Digi(process):
    if hasattr(process,'mix'):
        process.mix.digitizers.hcal.HBHEUpgradeQIE = True
        process.mix.digitizers.hcal.hb.photoelectronsToAnalog = cms.vdouble([10.]*16)
        process.mix.digitizers.hcal.hb.pixels = cms.int32(4500*4*2)
        process.mix.digitizers.hcal.he.photoelectronsToAnalog = cms.vdouble([10.]*16)
        process.mix.digitizers.hcal.he.pixels = cms.int32(4500*4*2)
        process.mix.digitizers.hcal.HFUpgradeQIE = True
        process.mix.digitizers.hcal.TestNumbering = True

    if hasattr(process,'simHcalDigis'):
        process.simHcalDigis.useConfigZSvalues=cms.int32(1)
        process.simHcalDigis.HBlevel=cms.int32(16)
        process.simHcalDigis.HElevel=cms.int32(16)
        process.simHcalDigis.HOlevel=cms.int32(16)
        process.simHcalDigis.HFlevel=cms.int32(16)

    process.digitisation_step.remove(process.simHcalTriggerPrimitiveDigis)
    process.digitisation_step.remove(process.simHcalTTPDigis)

    return process

def customise_Reco(process):
    # not sure why these are missing - but need to investigate later
    process.reconstruction_step.remove(process.castorreco)
    process.reconstruction_step.remove(process.CastorTowerReco)
    process.reconstruction_step.remove(process.ak7CastorJets)
    process.reconstruction_step.remove(process.ak7CastorJetID)
    return process

def customise_DQM(process):
    process.dqmoffline_step.remove(process.hcalDigiMonitor)
    process.dqmoffline_step.remove(process.hcalDeadCellMonitor)
    process.dqmoffline_step.remove(process.hcalBeamMonitor)
    process.dqmoffline_step.remove(process.hcalRecHitMonitor)
    process.dqmoffline_step.remove(process.hcalDetDiagNoiseMonitor)
    process.dqmoffline_step.remove(process.hcalNoiseMonitor)
    process.dqmoffline_step.remove(process.RecHitsDQMOffline)
    process.dqmoffline_step.remove(process.zdcMonitor)
    process.dqmoffline_step.remove(process.hcalMonitor)
    process.dqmoffline_step.remove(process.hcalHotCellMonitor)
    process.dqmoffline_step.remove(process.hcalRawDataMonitor)
    return process

def customise_harvesting(process):
    return process

def customise_Validation(process):
    process.validation_step.remove(process.AllHcalDigisValidation)
    process.validation_step.remove(process.RecHitsValidation)
    process.validation_step.remove(process.globalhitsanalyze)
    return process

def customise_condOverRides(process):
    return process
    
def customise_mixing(process):
    process.mixData.HBHEPileInputTag = cms.InputTag("simHcalUnsuppressedDigis")
    process.mixData.HOPileInputTag = cms.InputTag("simHcalUnsuppressedDigis")
    process.mixData.HFPileInputTag = cms.InputTag("simHcalUnsuppressedDigis")
    process.mixData.QIE10PileInputTag = cms.InputTag("simHcalUnsuppressedDigis","HFQIE10DigiCollection")
    process.mixData.QIE11PileInputTag = cms.InputTag("simHcalUnsuppressedDigis","HBHEQIE11DigiCollection")
    return process
