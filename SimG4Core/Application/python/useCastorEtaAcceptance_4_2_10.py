import FWCore.ParameterSet.Config as cms

def customise(process): # Using SL instead GFlash
    process.g4SimHits.Generator.MinEtaCut = cms.double(-7.0)
    process.g4SimHits.CastorSD.nonCompensationFactor = cms.double(0.77)

    return(process)
