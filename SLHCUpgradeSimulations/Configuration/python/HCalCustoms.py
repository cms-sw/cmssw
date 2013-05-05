import FWCore.ParameterSet.Config as cms

def customise_HcalPhase1(process):

    #common stuff
    process.load("CalibCalorimetry/HcalPlugins/Hcal_Conditions_forGlobalTag_cff")
    process.es_hardcode.toGet = cms.untracked.vstring(
                'GainWidths',
                'MCParams',
                'RecoParams',
                'RespCorrs',
                'QIEData',
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
                'CholeskyMatrices',
                'CovarianceMatrices'
                )
    
    process.es_hardcode.hcalTopologyConstants.mode=cms.string('HcalTopologyMode::SLHC')
    process.es_hardcode.hcalTopologyConstants.maxDepthHB=cms.int32(3)
    process.es_hardcode.hcalTopologyConstants.maxDepthHB=cms.int32(3)
    process.es_hardcode.hcalTopologyConstants.maxDepthHE=cms.int32(5)
    process.es_hardcode.HcalReLabel.RelabelHits=cms.untracked.bool(True)
    # Special Upgrade trick (if absent - regular case assumed)
    process.es_hardcode.GainWidthsForTrigPrims = cms.bool(True)
    
    process.hcalTopologyIdeal.hcalTopologyConstants.mode=cms.string('HcalTopologyMode::SLHC')
    process.hcalTopologyIdeal.hcalTopologyConstants.maxDepthHB=cms.int32(3)
    process.hcalTopologyIdeal.hcalTopologyConstants.maxDepthHE=cms.int32(5)
    

    if hasattr(process,'g4SimHits'):
        process=customise_Sim(process)
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process)
    process=customise_condOverRides(process)
    return process


def customise_Sim(process):
    process.g4SimHits.HCalSD.TestNumberingScheme = True

    return process

def customise_DigiToRaw(process):
    return process

def customise_RawToDigi(process):
    return process

def customise_Digi(process):
    if hasattr(process,'mix'):
        process.mix.digitizers.hcal.HBHEUpgradeQIE = True
        process.mix.digitizers.hcal.hb.siPMCells = cms.vint32([1])
        process.mix.digitizers.hcal.hb.photoelectronsToAnalog = cms.vdouble([10.]*16)
        process.mix.digitizers.hcal.hb.pixels = cms.int32(4500*4*2)
        process.mix.digitizers.hcal.he.photoelectronsToAnalog = cms.vdouble([10.]*16)
        process.mix.digitizers.hcal.he.pixels = cms.int32(4500*4*2)
        process.mix.digitizers.hcal.HFUpgradeQIE = True
        process.mix.digitizers.hcal.HcalReLabel.RelabelHits=cms.untracked.bool(True)

    if hasattr(process,'HcalTPGCoderULUT'):
        process.HcalTPGCoderULUT.hcalTopologyConstants.mode=cms.string('HcalTopologyMode::SLHC')
        process.HcalTPGCoderULUT.hcalTopologyConstants.maxDepthHB=cms.int32(3)
        process.HcalTPGCoderULUT.hcalTopologyConstants.maxDepthHE=cms.int32(5)

    process.digitisation_step.remove(process.simHcalTriggerPrimitiveDigis)
    process.digitisation_step.remove(process.simHcalTTPDigis)
    
    return process

def customise_Reco(process):
    return process

def customise_DQM(process):
    return process

def customise_harvesting(process):
    return process

def customise_Validation(process):
    return process

def customise_condOverRides(process):
    return process
