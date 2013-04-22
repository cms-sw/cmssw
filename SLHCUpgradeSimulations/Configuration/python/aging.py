import FWCore.ParameterSet.Config as cms

def customise_aging_300(process):

    #pixel rad dam recipe:
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/ExamplePhaseI#Pixel_Radiation_Damage_Studies
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.0)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(300)  # integrated lumi in fb-1
        process.g4SimHits.HCalSD.HEDarkening       = cms.untracked.bool(True)
        process.g4SimHits.HCalSD.HFDarkening       = cms.untracked.bool(True)
        #these lines need to be further activiated by tuning on 'complete' aging for ecal 
        process.g4SimHits.ECalSD.InstLuminosity = cms.double(1.0E34)
        process.g4SimHits.ECalSD.DelivLuminosity = cms.double(300.)


    return process


def customise_aging_500(process):

    #pixel rad dam recipe:
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/ExamplePhaseI#Pixel_Radiation_Damage_Studies
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.0)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(500)  # integrated lumi in fb-1
        process.g4SimHits.HCalSD.HEDarkening       = cms.untracked.bool(True)
        process.g4SimHits.HCalSD.HFDarkening       = cms.untracked.bool(True)
        #these lines need to be further activiated by tuning on 'complete' aging for ecal 
        process.g4SimHits.ECalSD.InstLuminosity = cms.double(1.0E34)
        process.g4SimHits.ECalSD.DelivLuminosity = cms.double(500.)

    return process

def customise_aging_1000(process):

    #pixel rad dam recipe:
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/ExamplePhaseI#Pixel_Radiation_Damage_Studies
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.0)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(3000)  # integrated lumi in fb-1
        process.g4SimHits.HCalSD.HEDarkening       = cms.untracked.bool(True)
        process.g4SimHits.HCalSD.HFDarkening       = cms.untracked.bool(True)
        #these lines need to be further activiated by tuning on 'complete' aging for ecal 
        process.g4SimHits.ECalSD.InstLuminosity = cms.double(1.0E34)
        process.g4SimHits.ECalSD.DelivLuminosity = cms.double(1000.)

    return process

def customise_aging_3000(process):

    #pixel rad dam recipe:
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/ExamplePhaseI#Pixel_Radiation_Damage_Studies
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.0)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(3000)  # integrated lumi in fb-1
        process.g4SimHits.HCalSD.HEDarkening       = cms.untracked.bool(True)
        process.g4SimHits.HCalSD.HFDarkening       = cms.untracked.bool(True)
        #these lines need to be further activiated by tuning on 'complete' aging for ecal 
        process.g4SimHits.ECalSD.InstLuminosity = cms.double(1.0E34)
        process.g4SimHits.ECalSD.DelivLuminosity = cms.double(3000.)

    return process
    
def ecal_complete_aging(proess):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.ECalSD.AgeingWithSlopeLY = cms.untracked.bool(True)
    if hasattr(process,ecal_digi_parameters):    
        process.ecal_digi_parameters.UseLCcorrection = cms.untracked.bool(False)
    return process

def turn_off_HE_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HHCalSD.HEDarkening       = cms.untracked.bool(False)
    return process

def turn_off_HF_aging(process):
    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.HFDarkening       = cms.untracked.bool(False)
    return process
