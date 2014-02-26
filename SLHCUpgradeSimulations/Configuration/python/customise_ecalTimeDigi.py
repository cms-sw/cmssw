import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.ecalTimeDigitizer_cfi import *

# Remove the Crossing Frames to save memory
def customise_addEcalTimeDigitizer(process):
    process.mix.digitizers.ecalTime=cms.PSet(
        ecalTimeDigitizer
      )
    print  process.mix.digitizers.parameters_()
    return (process)


    
