import FWCore.ParameterSet.Config as cms

def customise(process): # Using SL instead GFlash
    process.g4SimHits.HCalSD.UseShowerLibrary = cms.bool(True)
    process.g4SimHits.HCalSD.UseParametrize = cms.bool(False)
    process.g4SimHits.HCalSD.UsePMTHits = cms.bool(False)
    process.g4SimHits.HCalSD.UseFibreBundleHits = cms.bool(False)
    process.g4SimHits.HFShower.UseShowerLibrary = cms.bool(True)
    process.g4SimHits.HFShower.UseHFGflash = cms.bool(False)
    process.g4SimHits.HFShower.ApplyFiducialCut = cms.bool(False)

    return(process)
