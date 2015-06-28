import FWCore.ParameterSet.Config as cms

def customise_fastSimPostLS1(process):

    if hasattr(process,'famosSimHits'):
       process=customise_fastSimProducer(process)
       
    return process


def customise_fastSimProducer(process): 

    # enable 2015 HF shower library
    process.famosSimHits.Calorimetry.HFShowerLibrary.useShowerLibrary = True

    # change default parameters
    process.famosSimHits.ParticleFilter.pTMin  = 0.1
    process.famosSimHits.TrackerSimHits.pTmin  = 0.1
    process.famosSimHits.ParticleFilter.etaMax = 5.300

    return process


