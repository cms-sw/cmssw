import FWCore.ParameterSet.Config as cms

def customise_aging_300(process):

    #pixel rad dam recipe:
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/ExamplePhaseI#Pixel_Radiation_Damage_Studies
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.0)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(300)  # integrated lumi in fb-1
        process.g4SimHits.HCalSD.Darkening       = cms.untracked.bool(True)
               

    return process


def customise_aging_500(process):

    #pixel rad dam recipe:
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/ExamplePhaseI#Pixel_Radiation_Damage_Studies
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.0)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(500)  # integrated lumi in fb-1
        process.g4SimHits.HCalSD.Darkening       = cms.untracked.bool(True)

    return process

def customise_aging_1000(process):

    #pixel rad dam recipe:
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/ExamplePhaseI#Pixel_Radiation_Damage_Studies
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.0)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(3000)  # integrated lumi in fb-1
        process.g4SimHits.HCalSD.Darkening       = cms.untracked.bool(True)

    return process

def customise_aging_3000(process):

    #pixel rad dam recipe:
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/ExamplePhaseI#Pixel_Radiation_Damage_Studies
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.PseudoRadDamage =  cms.double(1.0)
        process.mix.digitizers.pixel.PseudoRadDamageRadius =  cms.double(4.0)

    if hasattr(process,'g4SimHits'):
        process.g4SimHits.HCalSD.DelivLuminosity = cms.double(3000)  # integrated lumi in fb-1
        process.g4SimHits.HCalSD.Darkening       = cms.untracked.bool(True)

    return process
    
