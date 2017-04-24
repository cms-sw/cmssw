import FWCore.ParameterSet.Config as cms

# handle normal mixing or premixing
def getHcalDigitizer(process):
    if hasattr(process,'mixData'):
        return process.mixData
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):
        return process.mix.digitizers.hcal
    return None

# turnon = True enables default, False disables
# recalibration and darkening always together
def ageHB(process,turnon):
    if turnon:
        from CalibCalorimetry.HcalPlugins.HBHEDarkening_cff import HBDarkeningEP
        process.HBDarkeningEP = HBDarkeningEP
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.HBDarkening = cms.bool(turnon)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HBRecalibration = cms.bool(turnon)
    return process

def ageHE(process,turnon):
    if turnon:
        from CalibCalorimetry.HcalPlugins.HBHEDarkening_cff import HEDarkeningEP
        process.HEDarkeningEP = HEDarkeningEP
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.HEDarkening = cms.bool(turnon)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HERecalibration = cms.bool(turnon)
    return process

def ageHF(process,turnon):
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.HFDarkening = cms.bool(turnon)
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.HFRecalibration = cms.bool(turnon)
    return process

# needs lumi to set proper ZS thresholds (tbd)
def ageSiPM(process,turnon,lumi):
    process.es_hardcode.hbUpgrade.doRadiationDamage = turnon
    process.es_hardcode.heUpgrade.doRadiationDamage = turnon

    # todo: determine ZS threshold adjustments

    return process

def ageHcal(process,lumi):
    instLumi=1.0e34
    if lumi>=1000:
        instLumi=5.0e34

    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.DelivLuminosity = cms.double(float(lumi))  # integrated lumi in fb-1

    # these lines need to be further activated by turning on 'complete' aging for HF 
    if hasattr(process,'g4SimHits'):  
        process.g4SimHits.HCalSD.InstLuminosity = cms.double(float(instLumi))
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(float(lumi))

    # recalibration and darkening always together
    if hasattr(process,'es_hardcode'):
        process.es_hardcode.iLumi = cms.double(float(lumi))

    # functions to enable individual subdet aging
    process = ageHB(process,True)
    process = ageHE(process,True)
    process = ageHF(process,True)
    process = ageSiPM(process,True,lumi)

    return process

def turn_on_HB_aging(process):
    process = ageHB(process,True)
    return process

def turn_off_HB_aging(process):
    process = ageHB(process,False)
    return process

def turn_on_HE_aging(process):
    process = ageHE(process,True)
    return process
    
def turn_off_HE_aging(process):
    process = ageHE(process,False)
    return process
    
def turn_on_HF_aging(process):
    process = ageHF(process,True)
    return process
    
def turn_off_HF_aging(process):
    process = ageHF(process,False)
    return process

def turn_off_SiPM_aging(process):
    process = ageSiPM(process,False,0.0)
    return process

def hf_complete_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.HFDarkening = cms.untracked.bool(True)
    hcaldigi = getHcalDigitizer(process)
    if hcaldigi is not None: hcaldigi.HFDarkening = cms.untracked.bool(False)
    return process

def ageEcal(process,lumi):

    instLumi=1.0e34
    if lumi>=1000:
        instLumi=5.0e34
        
    if hasattr(process,'g4SimHits'):
        #these lines need to be further activiated by tuning on 'complete' aging for ecal 
        process.g4SimHits.ECalSD.InstLuminosity = cms.double(instLumi)
        process.g4SimHits.ECalSD.DelivLuminosity = cms.double(float(lumi))
    return process

def customise_aging_300(process):

    process=ageHcal(process,300)
    process=ageEcal(process,300)
    return process

def customise_aging_1000(process):

    process=ageHcal(process,1000)
    process=ageEcal(process,1000)
    return process

def customise_aging_3000(process):

    process=ageHcal(process,3000)
    process=ageEcal(process,3000)
    return process

def ecal_complete_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.ECalSD.AgeingWithSlopeLY = cms.untracked.bool(True)
    if hasattr(process,'ecal_digi_parameters'):    
        process.ecal_digi_parameters.UseLCcorrection = cms.untracked.bool(False)
    return process
