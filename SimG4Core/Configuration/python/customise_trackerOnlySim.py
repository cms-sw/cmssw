import FWCore.ParameterSet.Config as cms

def customise(process):
    process.g4SimHits.OnlySDs = [
    'TkAccumulatingSensitiveDetector'
    ]
    return process
