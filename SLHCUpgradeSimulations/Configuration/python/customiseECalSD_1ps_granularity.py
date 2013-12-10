import FWCore.ParameterSet.Config as cms

#Store layer in 1 cm for ECAL and 1ps timeSlices
def customise(process):
    process.g4SimHits.ECalSD.StoreLayerTimeSim  = cms.untracked.bool(True)
    process.g4SimHits.ECalSD.TimeSliceUnit  = cms.double(0.001)
    return(process) 
