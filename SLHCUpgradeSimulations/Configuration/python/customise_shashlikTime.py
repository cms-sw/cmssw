import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.ecalTimeDigitizer_cfi import *

def cust_shashlikTime(process):
    # Store layers in 1 cm for Shashlik with 1ps timeSlices
    if hasattr(process,'g4SimHits'):
        print "#___ ShashlikSD configured for 1ps time resolution and 1cm layers ___"
        process.g4SimHits.ShashlikSD.StoreLayerTimeSim  = cms.untracked.bool(True)
        process.g4SimHits.ShashlikSD.TimeSliceUnit  = cms.double(0.001)

    # Switch on the ecalTime digitization
    if hasattr(process,'mix'):
        if ( hasattr( getattr( getattr( process, 'mix'), 'digitizers' ), 'ecal' ) ):
            print "#___ Adding ecalDetailedTime digitizer ___"
            process.mix.digitizers.ecalTime=cms.PSet(
                ecalTimeDigitizer
                )
    # Switch on the ecalDetailedTimeRecHit association step
    if hasattr(process,'reconstruction_step'):
        print "#___ Adding ecalDetailedTimeRecHit associator ___"
        process.load("RecoLocalCalo.EcalRecProducers.ecalDetailedTimeRecHit_cfi")
        process.reconstruction_step+=process.ecalDetailedTimeRecHit
        
    for out in process.outputModules_().iterkeys():
        dataTier=getattr( getattr ( getattr(process,out), 'dataset'), 'dataTier')
        if dataTier != 'DQM':
            print "#___ Adding ecalTimeOutputCommands to outputModule " + str(out) + " ___"
            outputCommands = getattr( getattr(process,out), 'outputCommands')
            ecalTimeOutputCommands=cms.untracked.vstring(
                'keep *_*_EBTimeDigi_*',
                'keep *_*_EETimeDigi_*', 
                'keep *_*_EKTimeDigi_*', 
                'keep *_ecalDetailedTimeRecHit_*_*')
            outputCommands.extend(ecalTimeOutputCommands)


    return(process)




    
