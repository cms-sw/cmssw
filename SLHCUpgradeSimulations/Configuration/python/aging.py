import FWCore.ParameterSet.Config as cms

def agePixel(process,lumi):
    prd=1.0
    if lumi==300:
        prd=1.0
    if lumi==500:
        prd=1.5
    if lumi==1000:
        prd=0.
    if lumi==3000:
        prd=0. #no aging yet
        
    # danger - watch for someone turning off pixel aging - if off - leave off
    if hasattr(process,'mix') and not hasattr(process.mix.digitizers.pixel,'NoAging'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(float(prd*7.65))
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(16.5)
    return process    

def ageHcal(process,lumi):

    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(float(lumi))  # integrated lumi in fb-1
        process.g4SimHits.HCalSD.HEDarkening       = cms.bool(True)
        process.g4SimHits.HCalSD.HFDarkening       = cms.bool(True)
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
    process=agePixel(process,300)
    return process

def customise_aging_500(process):

    process=ageHcal(process,500)
    process=ageEcal(process,500)
    process=agePixel(process,500)
    return process

def customise_aging_1000(process):

    process=ageHcal(process,1000)
    process=ageEcal(process,1000)
    process=agePixel(process,1000)
    return process

def customise_aging_3000(process):

    process=ageHcal(process,3000)
    process=ageEcal(process,3000)
    process=agePixel(process,3000)
    return process

def customise_aging_ecalonly_300(process):

    process=ageEcal(process,300)
    return process

def customise_aging_ecalonly_1000(process):

    process=ageEcal(process,1000)
    return process

def customise_aging_ecalonly_3000(process):

    process=ageEcal(process,3000)
    return process

def customise_aging_newpixel_1000(process):

    process=ageEcal(process,1000)
    process=ageHcal(process,1000)
    return process

def customise_aging_newpixel_3000(process):

    process=ageEcal(process,3000)
    process=ageHcal(process,3000)
    return process

#no hcal 3000

def ecal_complete_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.ECalSD.AgeingWithSlopeLY = cms.untracked.bool(True)
    if hasattr(process,'ecal_digi_parameters'):    
        process.ecal_digi_parameters.UseLCcorrection = cms.untracked.bool(False)
    return process

def turn_off_HE_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HHCalSD.HEDarkening       = cms.bool(False)
    return process

def turn_off_HF_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.HFDarkening       = cms.bool(False)
    return process

def turn_off_Pixel_aging(process):

    if hasattr(process,'mix'):
        setattr(process.mix.digitizers.pixel,'NoAging',cms.double(1.))
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(0.)
    return process

def turn_on_Pixel_aging_1000(process):
    # just incase we want aging afterall
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.5)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    return process


def reco_aging_hcal_stdgeom(process):
    process.load("CalibCalorimetry/HcalPlugins/Hcal_Conditions_forGlobalTag_cff")

    process.es_hardcode.HcalReLabel.RelabelHits = cms.untracked.bool(True)
    process.es_hardcode.HcalReLabel.RelabelRules.Eta1  =    cms.untracked.vint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    process.es_hardcode.HcalReLabel.RelabelRules.Eta16 =    cms.untracked.vint32(1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3)
    process.es_hardcode.HcalReLabel.RelabelRules.Eta17 =    cms.untracked.vint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    process.es_hardcode.HcalReLabel.RelabelRules.Eta18 =    cms.untracked.vint32(1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2)
    process.es_hardcode.HcalReLabel.RelabelRules.Eta19 =    cms.untracked.vint32(1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2)
    process.es_hardcode.HcalReLabel.RelabelRules.Eta27 =    cms.untracked.vint32(1,1,1,1,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3)

    process.es_hardcode.toGet = cms.untracked.vstring(
        'GainWidths',
        'RespCorrs'
        )
    process.es_hardcode.HERecalibration = cms.bool(True)
    process.es_hardcode.HFRecalibration = cms.bool(True)
    
    return process

def reco_aging_hcal_stdgeom_300(process):
    process=reco_aging_hcal_stdgeom(process)
    process.es_hardcode.iLumi = cms.double(300.)
    return process

def reco_aging_hcal_stdgeom_500(process):
    process=reco_aging_hcal_stdgeom(process)
    process.es_hardcode.iLumi = cms.double(500.)
    return process

def reco_aging_hcal_stdgeom_1000(process):
    process=reco_aging_hcal_stdgeom(process)
    process.es_hardcode.iLumi = cms.double(1000.)
    return process

def reco_aging_hcal_stdgeom_3000(process):
    process=reco_aging_hcal_stdgeom(process)
    process.es_hardcode.iLumi = cms.double(3000.)
    return process

